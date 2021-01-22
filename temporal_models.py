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
from torch.nn.utils import spectral_norm

FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(FlowSTSR_folder_path, "InputData")
output_folder = os.path.join(FlowSTSR_folder_path, "Output")
save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")


def train_temporal_network(model, discriminator, dataset, opt):
    model = model.to(opt['device'])
    discriminator = discriminator.to(opt['device'])

    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    generator_optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[8000-opt['iteration_number']],gamma=opt['gamma'])

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=opt["learning_rate"]*4, 
    betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_optimizer,
    milestones=[8000-opt['iteration_number']],gamma=opt['gamma'])

    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
    reference_writer = SummaryWriter(os.path.join('tensorboard', "LERP"))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=opt["num_workers"]
    )

    loss_function = nn.MSELoss().to(opt['device'])
    iters = 0
    for epoch in range(opt['epoch_number'], opt["epochs"]):        
        for batch_num, items in enumerate(dataloader):
            gt_start_frame = crop_to_size(items[0], opt['cropping_resolution']).to(opt['device'])
            gt_end_frame = crop_to_size(items[1], opt['cropping_resolution']).to(opt['device'])
            gt_middle_frames = crop_to_size(items[2][0], opt['cropping_resolution']).to(opt['device'])
            timesteps = (int(items[3][0]), int(items[3][1]))

            gt_start_frame = dataset.scale(gt_start_frame)
            gt_end_frame = dataset.scale(gt_end_frame)
            
            pred_frames = dataset.unscale(model(gt_start_frame, gt_end_frame, timesteps))

            for i in range(opt['discriminator_steps']):
                discriminator.zero_grad()
                discrim_loss = 0.5*torch.log(1-discriminator(pred_frames.detach())) + \
                0.5*torch.log(discriminator(gt_middle_frames))
                discrim_loss.backward()
                discriminator_optimizer.step()

            for i in range(opt['generator_steps']):
                generator.zero_grad()
                loss = loss_function(pred_frames, gt_middle_frames)
                loss.backward()
                gen_loss = torch.log(discriminator(pred_frames))
                gen_loss.backward()
                generator_optimizer.step()

            generator_scheduler.step()  
            discriminator_scheduler.step()

            pred_frame_cm_image = toImg(pred_frames[0].detach().cpu().numpy())
            gt_middle_frame_cm_image = toImg(gt_middle_frames[0].detach().cpu().numpy())
            err_frame_image = toImg(torch.abs((pred_frames[0].detach() - gt_middle_frames[0].detach())).cpu().numpy())

            lerped_frames = []
            for i in range(timesteps[1]-timesteps[0]-1):
                factor = (i+1)/(timesteps[1]-timesteps[0])
                lerped_gt = (1-factor)*gt_start_frame + \
                factor*gt_end_frame
                lerped_gt = lerped_frames.append(dataset.unscale(lerped_gt))
            lerped_gt = torch.cat(lerped_frames, dim=0)
            reference_writer.add_scalar('MSE', loss_function(lerped_gt, gt_middle_frames).item(), iters)
            writer.add_scalar('MSE', loss.item(), iters) 
            writer.add_scalar('D_loss', discrim_loss.item(), iters) 
            writer.add_scalar('G_loss', gen_loss.item(), iters) 
            writer.add_image("Predicted next frame",pred_frame_cm_image, iters)
            writer.add_image("GT next frame",gt_middle_frame_cm_image, iters)
            writer.add_image("Abs difference",err_frame_image, iters)

            print_to_log_and_console("%i/%i: MSE=%.06f" %
            (iters, opt['epochs']*len(dataset), loss.item()), 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

            iters += 1

        opt['epoch_number'] = epoch+1
        if(epoch % opt['save_every'] == 0):
            save_models(model, discriminator, opt)

    return model

def save_models(model, discriminator, opt):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        torch.save(model.state_dict(), os.path.join(path_to_save, "temporal_generator"))
    if(opt['save_discriminators']):
        torch.save(discriminator.state_dict(), os.path.join(path_to_save, "temporal_discriminator"))

    save_options(opt, path_to_save)

def load_models(model, discriminator, opt, device):
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    model.load_state_dict(torch.load(os.path.join(
        load_folder, "temporal_generator"), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(
        load_folder, "temporal_discriminator"), map_location=device))
    return model, discriminator

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm3d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)

def VoxelShuffle(t):
    # t has shape [batch, channels, x, y, z]
    # channels should be divisible by 8
    
    input_view = t.contiguous().view(
        1, 2, 2, 2, int(t.shape[1]/8), t.shape[2], t.shape[3], t.shape[4]
    )
    shuffle_out = input_view.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    out = shuffle_out.view(
        1, int(t.shape[1]/8), 2*t.shape[2], 2*t.shape[3], 2*t.shape[4]
    )
    return out


class Temporal_Discriminator(nn.Module):
    def __init__ (self, opt):
        super(Temporal_Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv3d(opt['num_channels'], 64, stride=2, kernel_size=4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(64, 128, stride=2, kernel_size=4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(128, 256, stride=2, kernel_size=4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 512, stride=2, kernel_size=4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(512, 1, stride=1, kernel_size=4))
        )

    def forward(self, x):
        return self.model(x).mean()

class Temporal_Generator(nn.Module):
    def __init__ (self, opt):
        super(Temporal_Generator, self).__init__()
        
        self.feature_learning = nn.Sequential(
            DownscaleBlock(opt['num_channels'], 16, 5, 2),
            DownscaleBlock(16, 32, 3, 1),
            DownscaleBlock(32, 64, 3, 1),
            DownscaleBlock(64, 64, 3, 1)
        )

        self.convlstm_forward = ConvLSTM(opt)
        self.convlstm_backward = ConvLSTM(opt)

        self.upscaling = nn.Sequential(
            UpscalingBlock(64, 64, 3, 1),
            UpscalingBlock(64, 32, 3, 1),
            UpscalingBlock(32, 16, 3, 1),
            UpscalingBlock(16, opt['num_channels'], 5, 2)
        )

        self.act = nn.Tanh()

    def forward(self, x_start, x_end, timesteps):
        '''
        x should be of shape (seq_length, c, x, y, z)
        timesteps should be (ts_start, ts_predicted, ts_end)
        '''
        pred_frames_forward = []
        pred_frames_backward = []
        pred_frames = []

        x_start_pred = x_start
        for i in range(timesteps[1]-timesteps[0]-1):
            x_start_pred = self.feature_learning(x_start_pred)
            x_start_pred = self.convlstm_forward(x_start_pred)
            x_start_pred = self.upscaling(x_start_pred)
            x_start_pred = self.act(x_start_pred)
            pred_frames_forward.append(x_start_pred)

        x_end_pred = x_end
        for i in range(timesteps[1]-timesteps[0]-1):
            x_end_pred = self.feature_learning(x_end_pred)
            x_end_pred = self.convlstm_backward(x_end_pred)
            x_end_pred = self.upscaling(x_end_pred)
            x_end_pred = self.act(x_end_pred)
            pred_frames_backward.insert(0, x_end_pred)

        for i in range(timesteps[1]-timesteps[0]-1):
            lerp_factor = float(i+1) / float(timesteps[1]-timesteps[0]+1)
            lerped_gt = (1.0-lerp_factor)*x_start + lerp_factor*x_end
            pred_frames.append(lerped_gt + 0.5*(pred_frames_forward[i] + pred_frames_backward[i]))
        
        pred_frames = torch.cat(pred_frames, dim=0)
        return pred_frames

class Temporal_Generator_UNET(nn.Module):
    def __init__ (self, opt):
        super(Temporal_Generator_UNET, self).__init__()
        self.opt = opt
        self.down1 = UNet_Downscaling_Module(opt['num_channels']*2, 64)
        self.down2 = UNet_Downscaling_Module(64, 128)
        self.down3 = UNet_Downscaling_Module(128, 256)

        self.up1 = DoubleConv(256, 512)
        self.up2 = DoubleConv(512+256, 256)
        self.up3 = DoubleConv(256+128, 128)
        self.up4 = DoubleConv(128+64, 64)

        self.finalconv = nn.Conv3d(64, opt['num_channels'], 
        kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_down1, x_through = self.down1(x)
        x_down2, x_through = self.down2(x_through)
        x_down3, x_through = self.down3(x_through)

        x_through = self.up1(x_through)

        x_through = F.interpolate(x_through, scale_factor=2, 
        mode=opt['upscaling_mode'], align_corners=True)
        x_through = self.up2(torch.cat([x_down3, x_through], dim=1))

        x_through = F.interpolate(x_through, scale_factor=2, 
        mode=opt['upscaling_mode'], align_corners=True)
        x_through = self.up3(torch.cat([x_down2, x_through], dim=1))

        x_through = F.interpolate(x_through, scale_factor=2, 
        mode=opt['upscaling_mode'], align_corners=True)
        x_through = self.up4(torch.cat([x_down1, x_through], dim=1))

        x_through = self.final_conv(x_through)

        return x_through
        


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet_Downscaling_Module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x2)
        return x1, x2

class DownscaleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(DownscaleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=2)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)           
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=2)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(input_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),        
        )
        
    def forward(self, x):
        return self.conv(x)

class UpscalingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(UpscalingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, output_channels*8, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(output_channels, output_channels, 
            kernel_size=kernel_size, padding=padding, stride=1)),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),       
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, output_channels*8, 
            kernel_size=1, padding=0, stride=1)),
            nn.InstanceNorm3d(output_channels*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        x1 = VoxelShuffle(self.conv1(x))
        x1 = self.conv2(x1)
        x2 = VoxelShuffle(self.conv3(x))
        return x1 + x2

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
        for filename in range(len(os.listdir(self.opt['data_folder']))):
            filename = str(filename) + ".h5"

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
            data_seq = (
                self.items[index],
                self.items[index+self.opt['training_seq_length']],                                 
                torch.stack(self.items[index+1:index+self.opt['training_seq_length']], dim=0),
                (index,index+self.opt['training_seq_length'])
            )            
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