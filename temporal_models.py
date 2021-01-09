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


class Generator(nn.Module):
    def __init__ (self, resolution, num_kernels, opt):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.opt = opt

        if(opt['physical_constraints'] == "hard" and (opt['mode'] == "2D" or opt['mode'] =="3Dto2D")):
            output_chans = 1
        else:
            output_chans = opt['num_channels']

        if(opt['pre_padding']):
            pad_amount = int(kernel_size/2)
            self.layer_padding = 0
        else:
            pad_amount = 0
            self.layer_padding = 1

        if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            self.required_padding = [pad_amount, pad_amount, pad_amount, pad_amount]
            self.upscale_method = "bicubic"
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d
            self.required_padding = [pad_amount, pad_amount, pad_amount, 
            pad_amount, pad_amount, pad_amount]
            self.upscale_method = "trilinear"

        if(not opt['separate_chans']):
            self.model = self.create_model(opt['num_blocks'], opt['num_channels'], output_chans,
            num_kernels, opt['kernel_size'], opt['stride'], 1,
            conv_layer, batchnorm_layer).to(opt['device'])
        else:
            self.model = self.create_model(opt['num_blocks'], opt['num_channels'], output_chans, 
            num_kernels, opt['kernel_size'], opt['stride'], opt['num_channels'],
            conv_layer, batchnorm_layer).to(opt['device'])

    def create_model(self, num_blocks, num_channels, output_chans,
    num_kernels, kernel_size, stride, groups, conv_layer, batchnorm_layer):
        modules = []
        
        for i in range(num_blocks):
            # The head goes from numChannels channels to numKernels
            if i == 0:
                modules.append(nn.Sequential(
                    conv_layer(num_channels, num_kernels*groups, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding, groups=groups),
                    batchnorm_layer(num_kernels*groups),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from kernel_size to num_channels before tanh [-1,1]
            elif i == num_blocks-1:  
                tail = nn.Sequential(
                    conv_layer(num_kernels*groups, output_chans, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding, groups=groups),
                    nn.Tanh()
                )              
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    conv_layer(num_kernels*groups, num_kernels*groups, kernel_size=kernel_size,
                    stride=stride, padding=self.layer_padding, groups=groups),
                    batchnorm_layer(num_kernels*groups),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        m = nn.Sequential(*modules)
        return m
        
    def get_input_shape(self):
        shape = []
        shape.append(1)
        shape.append(self.opt['num_channels'])
        for i in range(len(self.resolution)):
            shape.append(self.resolution[i])
        return shape

    def get_params(self):
        if(self.opt['separate_chans']):
            p = []
            for i in range(self.opt['num_channels']):
                p = p + list(self.model[i].parameters())
            return p
        else:
            return self.model.parameters()

    def receptive_field(self):
        return (self.opt['kernel_size']-1)*self.opt['num_blocks']

    def forward(self, data):
       
        if(self.opt['pre_padding']):
            data = F.pad(data, self.required_padding)
        output = self.model(data)

        if(self.opt['physical_constraints'] == "hard" and self.opt['mode'] == '3D'):
            output = curl3D(output, self.opt['device'])
            return output
        elif(self.opt['physical_constraints'] == "hard" and (self.opt['mode'] == '2D' or self.opt['mode'] == '3Dto2D')):
            output = curl2D(output, self.opt['device'])
            gradx = spatial_derivative2D(output[:,0:1], 0, self.opt['device'])
            grady = spatial_derivative2D(output[:,1:2], 1, self.opt['device'])
            output = torch.cat([-grady, gradx], axis=1)
            return output
        else:
            return output + data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None

        self.num_items = 0

        for filename in os.listdir(self.opt['data_folder']):
            d = np.load(os.path.join(self.opt['data_folder'], filename))

            print(filename + " " + str(d.shape))
            if(self.num_items == 0):
                self.num_channels = d.shape[0]
                self.resolution = d.shape[1:]
                if(self.opt['mode'] == "3Dto2D"):
                    self.resolution = self.resolution[0:len(self.resolution)-1]

            mags = np.linalg.norm(d, axis=0)
            m_mag = mags.max()
            if(self.max_mag is None or self.max_mag < m_mag):
                self.max_mag = m_mag

            for i in range(d.shape[0]):                
                if(len(self.channel_mins) <= i):
                    self.channel_mins.append(d[i].min())
                    self.channel_maxs.append(d[i].max())
                else:
                    if(d[i].max() > self.channel_maxs[i]):
                        self.channel_maxs[i] = d[i].max()
                    if(d[i].min() < self.channel_mins[i]):
                        self.channel_mins[i] = d[i].min()
            
            self.num_items += 1

    def __len__(self):
        return self.num_items

    def resolution(self):
        return self.resolution

    def scale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "magnitude"):
            d *= (1/self.max_mag)
        elif (self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[:,i] -= self.channel_mins[i]
                d[:,i] /= (self.channel_maxs[i] - self.channel_mins[i])
                d[:,i] -= 0.5
                d[:,i] *= 2
        return d

    def unscale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[0, i] *= 0.5
                d[0, i] += 0.5
                d[0, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[0, i] += self.channel_mins[i]
        elif(self.opt['scaling_mode'] == "magnitude"):
            d *= self.max_mag
        return d

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def __getitem__(self, index):
        data = np.load(os.path.join(self.opt['data_folder'], str(index) + ".npy"))
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                data[i] -= self.channel_mins[i]
                data[i] *= (1 / (self.channel_maxs[i] - self.channel_mins[i]))
                data[i] -= 0.5
                data[i] *= 2
        elif(self.opt['scaling_mode'] == "magnitude"):
            data *= (1 / self.max_mag)
            
        if(self.opt['mode'] == "3Dto2D"):
            data = data[:,:,:,int(data.shape[3]/2)]

        data = np2torch(data, "cpu")
        return data