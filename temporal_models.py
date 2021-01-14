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
import h5py

FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(FlowSTSR_folder_path, "InputData")
output_folder = os.path.join(FlowSTSR_folder_path, "Output")
save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")


def train_temporal_network(model, dataset, opt):
    model = model.to(opt['device'])

    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    generator_optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[8000-opt['iteration_number']],gamma=opt['gamma'])

    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=opt["num_workers"]
    )

    loss_function = nn.MSELoss().to(opt['device'])
    iters = 0
    for epoch in range(opt['epoch_number'], opt["epochs"]):        
        for batch_num, items in enumerate(dataloader):
            gt_frames = crop_to_size(items[0][0], opt['cropping_resolution']).to(opt['device'])
            gt_next_frame = crop_to_size(items[1], opt['cropping_resolution']).to(opt['device'])
            
            gt_frames = dataset.scale(gt_frames)
            
            pred_next_frame = dataset.unscale(model(gt_frames))
            loss = loss_function(pred_next_frame, gt_next_frame)
            loss.backward()
            
            generator_optimizer.step()
            generator_scheduler.step()  

            pred_next_frame_cm_image = toImg(pred_next_frame[0].detach().cpu().numpy())
            gt_next_frame_cm_image = toImg(gt_next_frame[0].detach().cpu().numpy())
            
            writer.add_scalar('MSE', loss.item(), iters) 
            writer.add_image("Predicted next frame",pred_next_frame_cm_image, iters)
            writer.add_image("GT next frame",gt_next_frame_cm_image, iters)

            print_to_log_and_console("%i/%i: MSE=%.02f" %
            (iters, opt['epochs']*len(dataset), loss.item()), 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

            iters += 1

    return model

def save_model(model, opt):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        torch.save(generator.state_dict(), os.path.join(path_to_save, "temporal_generator"))

    save_options(opt, path_to_save)

def load_model(model, opt, device):
    generators = []
    discriminators = []
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    model.load_state_dict(torch.load(os.path.join(
        load_folder, "temporal_generator"), map_location=device))
    return model

class Temporal_Generator(nn.Module):
    def __init__ (self, opt):
        super(Temporal_Generator, self).__init__()
        
        self.feature_learning = nn.Sequential(
            DownscaleBlock(opt['num_channels'], 16, 5, 2),
            DownscaleBlock(16, 32, 3, 1),
            DownscaleBlock(32, 64, 3, 1),
            DownscaleBlock(64, 64, 3, 1)
        )
        self.convlstm = ConvLSTM(opt)

        self.upscaling = nn.Sequential(
            UpscalingBlock(64, 64*8, 3, 1),
            UpscalingBlock(64, 64*8, 3, 1),
            UpscalingBlock(64, 32*8, 3, 1),
            UpscalingBlock(32, 32*8, 5, 2)
        )

        self.act = nn.Tanh()

    def forward(self, x):
        '''
        x should be of shape (seq_length, c, x, y, z)
        '''
        x = self.feature_learning(x)
        x = self.convlstm(x)
        x = self.upscaling(x)
        x = self.act(x)
        return x

class DownscaleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(DownscaleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=2)),
            nn.ReLU()            
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=2))
        )
        
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU()            
        )
        
    def forward(self, x):
        return self.conv(x)

class UpscalingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(UpscalingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channels, input_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(input_channels, input_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(input_channels, input_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.ReLU()            
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1))
        )
        
    def forward(self, x):
        return VoxelShuffle(self.conv1(x) + self.conv2(x))

def VoxelShuffle(t):
    # t has shape [batch, channels, x, y, z]
    # channels should be divisible by 8
    
    '''
    shape = list(t.shape)
    shape[1] = int(shape[1] / 8)
    shape[2] = shape[2] * 2
    shape[3] = shape[3] * 2
    shape[4] = shape[4] * 2

    a = torch.empty(shape).to(t.device)
    a.requires_grad = t.requires_grad
    a[:,:,::2,::2,::2] = t[:,0::8,:,:,:]
    a[:,:,::2,::2,1::2] = t[:,1::8,:,:,:]
    a[:,:,::2,1::2,::2] = t[:,2::8,:,:,:]
    a[:,:,::2,1::2,1::2] = t[:,3::8,:,:,:]
    a[:,:,1::2,::2,::2] = t[:,4::8,:,:,:]
    a[:,:,1::2,::2,1::2] = t[:,5::8,:,:,:]
    a[:,:,1::2,1::2,::2] = t[:,6::8,:,:,:]
    a[:,:,1::2,1::2,1::2] = t[:,7::8,:,:,:]
    #return a
    '''
    input_view = t.contiguous().view(
        1, 2, 2, 2, int(t.shape[1]/8), t.shape[2], t.shape[3], t.shape[4]
    )
    shuffle_out = input_view.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    out = shuffle_out.view(
        1, int(t.shape[1]/8), 2*t.shape[2], 2*t.shape[3], 2*t.shape[4]
    )
    return out

class ConvLSTMCell(nn.Module):
    def __init__(self, opt):

        super(ConvLSTMCell, self).__init__()

        self.opt = opt

        self.conv = nn.Conv3d(in_channels=64*2,
                              out_channels=64*8,
                              kernel_size=3,
                              padding=1, 
                              groups=2)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        i_x, f_x, o_x, c_x, i_h, f_h, o_h, c_h = torch.chunk(combined_conv, 
        8, dim=1)
        
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        o = torch.sigmoid(o_x + o_h)
        
        c_next = f * c_cur + i * torch.tanh(c_x + c_h)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, shape):
        return (torch.zeros(1, shape[1], shape[2], shape[3], shape[4], device=self.opt['device']),
                torch.zeros(1, shape[1], shape[2], shape[3], shape[4], device=self.opt['device']))

class ConvLSTM(nn.Module):
    def __init__(self, opt):
        super(ConvLSTM, self).__init__()
        self.opt = opt

        cell_list = []
        for i in range(0, opt['num_lstm_layers']):
            cell_list.append(ConvLSTMCell(opt))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        seq_length, ch, h, w, d = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(input_tensor.shape)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.opt['num_lstm_layers']):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_length):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t:t+1, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.cat(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output[-1,:,:,:,:].unsqueeze(0)

    def _init_hidden(self, shape):
        init_states = []
        for i in range(self.opt['num_lstm_layers']):
            init_states.append(self.cell_list[i].init_hidden(shape))
        return init_states

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):

            if(opt['load_data_at_start'] or (self.num_items > 0 and \
            (opt['scaling_mode'] == "magnitude" or opt['scaling_mode'] == "channel"))):
                print("Loading " + filename)   
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()

            if(self.num_items == 0):                             
                self.num_channels = d.shape[0]
                self.resolution = d.shape[1:]
                if(self.opt['mode'] == "3Dto2D"):
                    self.resolution = self.resolution[0:len(self.resolution)-1]

            if(opt['load_data_at_start']):
                self.items.append(d)

            if(opt['scaling_mode'] == "magnitude"):  
                mags = torch.norm(d, dim=0)
                m_mag = mags.max()
                if(self.max_mag is None or self.max_mag < m_mag):
                    self.max_mag = m_mag

            if(opt['scaling_mode'] == "channel"):
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
        return self.num_items - self.opt['training_seq_length']

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
                d[:, i] *= 0.5
                d[:, i] += 0.5
                d[:, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[:, i] += self.channel_mins[i]
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
        if(self.opt['load_data_at_start']):
            if(self.opt['temporal_direction'] == "forward"):
                data_seq = (torch.stack(self.items[index:index+self.opt['training_seq_length']], dim=0), 
                self.items[index+self.opt['training_seq_length']]) 
            elif(self.opt['temporal_direction'] == "backward"):
                data_seq = (torch.stack(self.items[index+1:index+self.opt['training_seq_length']+1][::-1], dim=0), 
                self.items[index]) 
        else:
            print("trying to load " + str(index) + ".h5")
            f = h5py.File(os.path.join(self.opt['data_folder'], str(index)+".h5"), 'r')
            print("converting " + str(index) + ".h5 to tensor")
            data =  torch.tensor(f.get('data'))
            f.close()
            print("converted " + str(index) + ".h5 to tensor")

        '''
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                data[i] -= self.channel_mins[i]
                data[i] *= (1 / (self.channel_maxs[i] - self.channel_mins[i]))
                data[i] -= 0.5
                data[i] *= 2
        elif(self.opt['scaling_mode'] == "magnitude"):
            data *= (1 / self.max_mag)
        '''

        if(self.opt['mode'] == "3Dto2D"):
            data = data[:,:,:,int(data.shape[3]/2)]

        #data = np2torch(data, "cpu")
        #print("returning " + str(index) + " data")
        
        return data_seq