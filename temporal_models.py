from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import time
import math
import random
import datetime
import os
from utility_functions import *
from options import *
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
from matplotlib.pyplot import cm
from math import pi
from skimage.transform.pyramids import pyramid_reduce
from torch.utils.tensorboard import SummaryWriter
import copy
from pytorch_memlab import LineProfiler, MemReporter, profile, profile_every

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")


class Temporal_Generator(nn.Module):
    def __init__ (self, opt):
        super(Temporal_Generator, self).__init__()
        
        self.resBlock1 = ResidualBlock(opt['num_channels'], 16, 5, 1)
        self.resBlock2 = ResidualBlock(16, 32, 3, 1)
        self.resBlock3 = ResidualBlock(32, 64, 3, 1)
        self.resBlock4 = ResidualBlock(64, 64, 3, 1)

        self.convlstm = ConvLSTM(64, [64], 3)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        self.block = [
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3D(input_channels, output_channels, 
                kernel_size=kernel_size, padding=padding, stride=1)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Conv3D(output_channels, output_channels, 
                kernel_size=kernel_size, padding=padding, stride=1)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Conv3D(output_channels, output_channels, 
                kernel_size=kernel_size, padding=padding, stride=1)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Conv3D(output_channels, output_channels, 
                kernel_size=kernel_size, padding=padding, stride=2)),
                nn.ReLU()            
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3D(input_channels, output_channels, 
                kernel_size=kernel_size, padding=padding, stride=2)))
            )
        ]

    def forward(self, x):
        return self.block[0](x) + self.block[1](x)


'''
From https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
'''
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w_d, kernel_size, opt, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv3d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width, self._state_depth = b_h_w_d
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, self._state_depth)).to(opt['device'])
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, self._state_depth)).to(opt['device'])
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, self._state_depth)).to(opt['device'])
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=cfg.HKO.BENCHMARK.IN_LEN):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width, depth = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width, depth))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)