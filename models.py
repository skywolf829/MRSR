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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def TAD(field, device):
    if(field.shape[1] == 2):
        tx = spatial_derivative2D(field[:,0:1,:,:], 1, device)
        ty = spatial_derivative2D(field[:,1:2,:,:], 0, device)
        g = torch.abs(tx + ty)
    elif(field.shape[1] == 3):
        tx = spatial_derivative2D(field[:,0:1,:,:], 1, device)
        ty = spatial_derivative2D(field[:,1:2,:,:], 0, device)
        g = torch.abs(tx + ty)
    return g

def TAD3D(field, device):
    tx = spatial_derivative3D(field[:,0:1,:,:,:], 2, device)
    ty = spatial_derivative3D(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D(field[:,2:3,:,:,:], 0, device)
    g = torch.abs(tx + ty + tz)
    return g

def curl2D(field, device):
    dydx = spatial_derivative2D(field[:,1:2], 0, device)
    dxdy = spatial_derivative2D(field[:,0:1], 1, device)
    output = dydx-dxdy
    return output

def curl3D(field, device):
    dzdy = spatial_derivative3D_CD(field[:,2:3], 1, device)
    dydz = spatial_derivative3D_CD(field[:,1:2], 2, device)
    dxdz = spatial_derivative3D_CD(field[:,0:1], 2, device)
    dzdx = spatial_derivative3D_CD(field[:,2:3], 0, device)
    dydx = spatial_derivative3D_CD(field[:,1:2], 0, device)
    dxdy = spatial_derivative3D_CD(field[:,0:1], 1, device)
    output = torch.cat((dzdy-dydz,dxdz-dzdx,dydx-dxdy), 1)
    return output

def curl3D8(field, device):
    dzdy = spatial_derivative3D_CD8(field[:,2:3], 1, device)
    dydz = spatial_derivative3D_CD8(field[:,1:2], 2, device)
    dxdz = spatial_derivative3D_CD8(field[:,0:1], 2, device)
    dzdx = spatial_derivative3D_CD8(field[:,2:3], 0, device)
    dydx = spatial_derivative3D_CD8(field[:,1:2], 0, device)
    dxdy = spatial_derivative3D_CD8(field[:,0:1], 1, device)
    output = torch.cat((dzdy-dydz,dxdz-dzdx,dydx-dxdy), 1)
    return output

def TAD3D_CD(field, device):
    tx = spatial_derivative3D_CD(field[:,0:1,:,:,:], 0, device)
    ty = spatial_derivative3D_CD(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D_CD(field[:,2:3,:,:,:], 2, device)
    g = torch.abs(tx + ty + tz)
    return g

def TAD3D_CD8(field, device):
    tx = spatial_derivative3D_CD8(field[:,0:1,:,:,:], 0, device)
    ty = spatial_derivative3D_CD8(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D_CD8(field[:,2:3,:,:,:], 2, device)
    g = torch.abs(tx + ty + tz)
    return g

def spatial_derivative2D_sobel(field, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == 0):
        weights = torch.tensor(
            np.array([
            [-1/8, 0, 1/8], 
            [-1/4, 0, 1/4],
            [-1/8, 0, 1/8]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    elif(axis == 1):
        weights = torch.tensor(
            np.array([
            [-1/8, -1/4, -1/8], 
            [   0,    0,    0], 
            [ 1/8,  1/4,  1/8]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    return output

def spatial_derivative2D(field, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == 0):
        weights = torch.tensor(
            np.array([
            [0, 0, 0], 
            [-0.5, 0, 0.5],
            [0, 0, 0]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    elif(axis == 1):
        weights = torch.tensor(
            np.array([
            [0, -0.5, 0], 
            [0,  0,   0], 
            [0, 0.5,  0]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    return output

def spatial_derivative3D_CD(field, axis, device):
    m = nn.ReplicationPad3d(1)
    # the first (a) axis in [a, b, c]
    if(axis == 0):
        weights = torch.tensor(np.array(
            [[[0, 0, 0], 
            [0, -0.5, 0],
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0.5, 0], 
            [0, 0, 0]]])
            .astype(np.float32)).to(device)
    elif(axis == 1):        
        # the second (b) axis in [a, b, c]
        weights = torch.tensor(np.array([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, -0.5, 0], 
            [0, 0, 0], 
            [0, 0.5, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]]])
            .astype(np.float32)).to(device)
    elif(axis == 2):
        # the third (c) axis in [a, b, c]
        weights = torch.tensor(np.array([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [-0.5, 0, 0.5], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0,  0], 
            [ 0, 0, 0]]])
            .astype(np.float32)).to(device)
    weights = weights.view(1, 1, 3, 3, 3)
    field = m(field)
    output = F.conv3d(field, weights)
    return output

def spatial_derivative3D_CD8(field, axis, device):
    m = nn.ReplicationPad3d(4)
    # the first (a) axis in [a, b, c]
    if(axis == 0):
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[0, 4, 4] = 1/280
        weights[1, 4, 4] = -4/105
        weights[2, 4, 4] = 1/5
        weights[3, 4, 4] = -4/5
        weights[4, 4, 4] = 0
        weights[5, 4, 4] = 4/5
        weights[6, 4, 4] = -1/5
        weights[7, 4, 4] = 4/105
        weights[8, 4, 4] = -1/280
        
    elif(axis == 1):        
        # the second (b) axis in [a, b, c]
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[4, 0, 4] = 1/280
        weights[4, 1, 4] = -4/105
        weights[4, 2, 4] = 1/5
        weights[4, 3, 4] = -4/5
        weights[4, 4, 4] = 0
        weights[4, 5, 4] = 4/5
        weights[4, 6, 4] = -1/5
        weights[4, 7, 4] = 4/105
        weights[4, 8, 4] = -1/280
    elif(axis == 2):
        # the third (c) axis in [a, b, c]
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[4, 4, 1] = 1/280
        weights[4, 4, 1] = -4/105
        weights[4, 4, 2] = 1/5
        weights[4, 4, 3] = -4/5
        weights[4, 4, 4] = 0
        weights[4, 4, 5] = 4/5
        weights[4, 4, 6] = -1/5
        weights[4, 4, 7] = 4/105
        weights[4, 4, 8] = -1/280
    weights = weights.view(1, 1, 9, 9, 9)
    field = m(field)
    output = F.conv3d(field, weights)
    return output

def calc_gradient_penalty(discrim, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discrim(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def mag_difference(t1, t2):
    mag_1 = torch.zeros(t1.shape).to(t1.device)
    mag_2 = torch.zeros(t1.shape).to(t1.device)
    for i in range(t1.shape[1]):
        mag_1[0, 0] += t1[0, i]**2
        mag_2[0, 0] += t2[0, i]**2
    
    mag_1 = torch.sqrt(mag_1[0:, 0:1])
    mag_2 = torch.sqrt(mag_2[0:, 0:1])
    mag_diff = torch.abs(mag_2-mag_1)
    '''
    t_1 = t1*(1/torch.norm(t1, dim=1).view(1, 1, t1.shape[2], t1.shape[3]).repeat(1, t1.shape[1], 1, 1))
    t_2 = t2*(1/torch.norm(t2, dim=1).view(1, 1, t1.shape[2], t1.shape[3]).repeat(1, t1.shape[1], 1, 1))
    c = (t_1* t_2).sum(dim=1)

    angle_diff = torch.acos(c)
    angle_diff[angle_diff != angle_diff] = 0
    angle_diff = angle_diff.unsqueeze(0)    
    '''
    return mag_diff

def reflection_pad2D(frame, padding, device):
    frame = F.pad(frame, 
    [padding, padding, padding, padding])
    indices_to_fix = []
    for i in range(0, padding):
        indices_to_fix.append(i)
    for i in range(frame.shape[2] - padding, frame.shape[2]):
        indices_to_fix.append(i)

    for x in indices_to_fix:
        if(x < padding):
            correct_x = frame.shape[2] - 2*padding - x
        else:
            correct_x = x - frame.shape[2] + 2*padding
        for y in indices_to_fix:
            if(y < padding):
                correct_y = frame.shape[3] - 2*padding - y
            else:
                correct_y = y - frame.shape[3] + 2*padding
            frame[:, :, x, y] = frame[:, :, correct_x, correct_y]
    return frame

def reflection_pad3D(frame, padding, device):
    frame = F.pad(frame, 
    [padding, padding, padding, padding, padding, padding])
    indices_to_fix = []
    for i in range(0, padding):
        indices_to_fix.append(i)
    for i in range(frame.shape[2] - padding, frame.shape[2]):
        indices_to_fix.append(i)
    for x in indices_to_fix:
        if(x < padding):
            correct_x = frame.shape[2] - 2*padding - x
        else:
            correct_x = x - frame.shape[2] + 2*padding
        for y in indices_to_fix:
            if(y < padding):
                correct_y = frame.shape[3] - 2*padding - y
            else:
                correct_y = y - frame.shape[3] + 2*padding
            for z in indices_to_fix:
                if(z < padding):
                    correct_z = frame.shape[4] - 2*padding - z
                else:
                    correct_z = z - frame.shape[4] + 2*padding
                frame[:, :, x, y, z] = frame[:, :, correct_x, correct_y, correct_z]
    return frame

def laplace_pyramid_downscale2D(frame, level, downscale_per_level, device, periodic=False):
    kernel_size = 5
    sigma = 2 * (1 / downscale_per_level) / 6

    xy_grid = torch.zeros([kernel_size, kernel_size, 2])
    for i in range(kernel_size):
        for j in range(kernel_size):
                xy_grid[i, j, 0] = i
                xy_grid[i, j, 1] = j

    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(frame.shape[1], 1, 1, 1)
    input_size = np.array(list(frame.shape[2:]))
    with torch.no_grad():
        for i in range(level):
            s = (input_size * (downscale_per_level**(i+1))).astype(int)
            if(periodic):
                frame = reflection_pad2D(frame, int(kernel_size / 2), device)
            
            frame = F.conv2d(frame, gaussian_kernel, groups=frame.shape[1])
            frame = F.interpolate(frame, size = list(s), mode='bilinear', align_corners=False)
    del gaussian_kernel
    return frame

def laplace_pyramid_downscale3D(frame, level, downscale_per_level, device, periodic=False):
    kernel_size = 5
    sigma = 2 * (1 / downscale_per_level) / 6

    xyz_grid = torch.zeros([kernel_size, kernel_size, kernel_size, 3])
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                xyz_grid[i, j, k, 0] = i
                xyz_grid[i, j, k, 1] = j
                xyz_grid[i, j, k, 2] = k
   
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xyz_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, 
    kernel_size, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(frame.shape[1], 1, 1, 1, 1)
    input_size = np.array(list(frame.shape[2:]))
    
    with torch.no_grad():
        for i in range(level):
            s = (input_size * (downscale_per_level**(i+1))).astype(int)
            if(periodic):
                frame = reflection_pad3D(frame, int(kernel_size / 2), device)
            frame = F.conv3d(frame, gaussian_kernel, groups=frame.shape[1])
            frame = F.interpolate(frame, size = list(s), mode='trilinear', align_corners=False)
    del gaussian_kernel
    return frame

def downsample(input_frame, output_size, downsample_mode):
    frame = F.interpolate(input_frame, size = output_size, mode=downsample_mode, align_corners=False)
    return frame

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def generate_padded_noise(size, pad_size, pad_with_noise, mode, device):
    if(pad_with_noise):
        for i in range(2,len(size)):
            size[i] += 2*pad_size
        noise = torch.randn(size, device=device)
    else:
        noise = torch.randn(size, device=device)
        if mode == "2D":
            required_padding = [pad_size, pad_size, pad_size, pad_size]
        else:
            required_padding = [pad_size, pad_size, pad_size, pad_size, pad_size, pad_size]
        noise = F.pad(noise, required_padding)
    return noise

def init_scales(opt, dataset):
    ns = []
    
    if(opt["spatial_downscale_ratio"] < 1.0):
        for i in range(len(dataset.resolution)):
            ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[i]) / math.log(opt["spatial_downscale_ratio"])))

    opt["n"] = min(ns)
    print("The model will have %i generators" % (opt["n"]))
    for i in range(opt["n"]+1):
        scaling = []
        factor =  opt["spatial_downscale_ratio"]**i
        for j in range(len(dataset.resolution)):
            x = int(dataset.resolution[j] * factor)
            scaling.append(x)
        opt["resolutions"].insert(0,scaling)
    for i in range(opt['n']):
        print("Scale %i: %s -> %s" % (opt["n"] - 1 - i, str(opt["resolutions"][i]), str(opt["resolutions"][i+1])))

def init_gen(scale, opt):
    num_kernels = int( 2** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    generator = Generator(opt["resolutions"][scale+1], num_kernels, opt)
    generator.apply(weights_init)

    return generator, num_kernels

def init_discrim(scale, opt):
    num_kernels = int(2 ** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    discriminator = Discriminator(opt["resolutions"][scale+1], num_kernels, opt)
    discriminator.apply(weights_init)

    return discriminator

def generate(generators, opt, starting_volume, start_scale=0):
    with torch.no_grad():
        
        for i in range(start_scale, len(generators)):
            starting_volume = F.interpolate(starting_volume, 
            size=opt["cropping_resolution"], mode=opt["upsample_mode"], align_corners=False)
            upscaled_volume = generators[i](starting_volume)

    return upscaled_volume

def generate_by_patch(generators, mode, opt, device, patch_size, 
generated_volume=None, start_scale=0):
    with torch.no_grad():
        #seq = []
        if(generated_volume is None):
            generated_volume = torch.zeros(generators[0].get_input_shape()).to(device)
        
        for i in range(start_scale, len(generators)):
            #print("Gen " + str(i))
            rf = int(generators[i].receptive_field() / 2)
            
            LR = F.interpolate(generated_volume, 
            size=generators[i].resolution, mode=opt["upsample_mode"], align_corners=False)
            generated_volume = torch.zeros(generators[i].get_input_shape()).to(device)

            if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
                y_done = False
                y = 0
                y_stop = min(generated_volume.shape[2], y + patch_size)
                while(not y_done):
                    if(y_stop == generated_volume.shape[2]):
                        y_done = True
                    x_done = False
                    x = 0
                    x_stop = min(generated_volume.shape[3], x + patch_size)
                    while(not x_done):                        
                        if(x_stop == generated_volume.shape[3]):
                            x_done = True

                        noise = full_noise[:,:,y:y_stop,x:x_stop]

                        #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                        result = generators[i](LR[:,:,y:y_stop,x:x_stop], 
                        opt["noise_amplitudes"][i]*noise)

                        x_offset = rf if x > 0 else 0
                        y_offset = rf if y > 0 else 0

                        generated_volume[:,:,
                        y+y_offset:y+noise.shape[2],
                        x+x_offset:x+noise.shape[3]] = result[:,:,y_offset:,x_offset:]

                        x += patch_size - 2*rf
                        x = min(x, max(0, generated_volume.shape[3] - patch_size))
                        x_stop = min(generated_volume.shape[3], x + patch_size)
                    y += patch_size - 2*rf
                    y = min(y, max(0, generated_volume.shape[2] - patch_size))
                    y_stop = min(generated_volume.shape[2], y + patch_size)

        
            elif(opt['mode'] == '3D'):
                
                z_done = False
                z = 0
                z_stop = min(generated_volume.shape[2], z + patch_size)
                while(not z_done):
                    if(z_stop == generated_volume.shape[2]):
                        z_done = True
                    y_done = False
                    y = 0
                    y_stop = min(generated_volume.shape[3], y + patch_size)
                    while(not y_done):
                        if(y_stop == generated_volume.shape[3]):
                            y_done = True
                        x_done = False
                        x = 0
                        x_stop = min(generated_volume.shape[4], x + patch_size)
                        while(not x_done):                        
                            if(x_stop == generated_volume.shape[4]):
                                x_done = True

                            noise = full_noise[:,:,z:z_stop,y:y_stop,x:x_stop]

                            #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                            result = generators[i](LR[:,:,z:z_stop,y:y_stop,x:x_stop], 
                            opt["noise_amplitudes"][i]*noise)

                            x_offset = rf if x > 0 else 0
                            y_offset = rf if y > 0 else 0
                            z_offset = rf if z > 0 else 0

                            generated_volume[:,:,
                            z+z_offset:z+noise.shape[2],
                            y+y_offset:y+noise.shape[3],
                            x+x_offset:x+noise.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                            x += patch_size - 2*rf
                            x = min(x, max(0, generated_volume.shape[4] - patch_size))
                            x_stop = min(generated_volume.shape[4], x + patch_size)
                        y += patch_size - 2*rf
                        y = min(y, max(0, generated_volume.shape[3] - patch_size))
                        y_stop = min(generated_volume.shape[3], y + patch_size)
                    z += patch_size - 2*rf
                    z = min(z, max(0, generated_volume.shape[2] - patch_size))
                    z_stop = min(generated_volume.shape[2], z + patch_size)


    #seq.append(generated_volume.detach().cpu().numpy()[0].swapaxes(0,2).swapaxes(0,1))
    #seq = np.array(seq)
    #seq -= seq.min()
    #seq /= seq.max()
    #seq *= 255
    #seq = seq.astype(np.uint8)
    #imageio.mimwrite("patches_good.gif", seq)
    #imageio.imwrite("patch_good_ex0.png", seq[0,0:100, 0:100,:])
    #imageio.imwrite("patch_good_ex1.png", seq[1,0:100, 0:100,:])
    #imageio.imwrite("patch_good_ex2.png", seq[2,0:100, 0:100,:])
    return generated_volume

def super_resolution(generator, frame, factor, opt, device):
    
    frame = frame.to(device)
    full_size = list(frame.shape[2:])
    for i in range(len(full_size)):
        full_size[i] *= factor
    r = 1 / opt["spatial_downscale_ratio"]
    curr_r = 1.0
    while(curr_r * r < factor):
        frame = F.interpolate(frame, scale_factor=r,mode=opt["upsample_mode"], align_corners=False)
        noise = torch.randn(frame.shape).to(device)
        frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
        curr_r *= r
    frame = F.interpolate(frame, size=full_size, mode=opt["upsample_mode"], align_corners=False)
    noise = torch.randn(frame.shape).to(device)
    noise = torch.zeros(frame.shape).to(device)
    frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
    return frame

def save_models(generators, discriminators, opt, optimizer=None):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        gen_states = {}
        
        for i in range(len(generators)):
            gen_states[str(i)] = generators[i].state_dict()
        torch.save(gen_states, os.path.join(path_to_save, "generators"))

    if(opt["save_discriminators"]):
        discrim_states = {}
        for i in range(len(discriminators)):
            discrim_states[str(i)] = discriminators[i].state_dict()
        torch.save(discrim_states, os.path.join(path_to_save, "discriminators"))

    save_options(opt, path_to_save)

def load_models(opt, device):
    generators = []
    discriminators = []
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(load_folder, "generators")):
        gen_params = torch.load(os.path.join(load_folder, "generators"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in gen_params.keys()):
                gen_params_compat = OrderedDict()
                for k, v in gen_params[str(i)].items():
                    if("module" in k):
                        gen_params_compat[k[7:]] = v
                    else:
                        gen_params_compat[k] = v
                generator, num_kernels = init_gen(i, opt)
                generator.load_state_dict(gen_params_compat)
                generators.append(generator)

        print_to_log_and_console("Successfully loaded generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if os.path.exists(os.path.join(load_folder, "discriminators")):
        discrim_params = torch.load(os.path.join(load_folder, "discriminators"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in discrim_params.keys()):
                discrim_params_compat = OrderedDict()
                for k, v in discrim_params[str(i)].items():
                    if(k[0:7] == "module."):
                        discrim_params_compat[k[7:]] = v
                    else:
                        discrim_params_compat[k] = v
                discriminator = init_discrim(i, opt)
                discriminator.load_state_dict(discrim_params_compat)
                discriminators.append(discriminator)
        print_to_log_and_console("Successfully loaded discriminators", 
        os.path.join(opt["save_folder"],opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "s_discriminators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    return  generators, discriminators

def train_single_scale_wrapper(generators, discriminators, opt):
    with LineProfiler(train_single_scale, generate, generate_by_patch, Generator.forward) as prof:
        g, d = train_single_scale(generators, discriminators, opt)
    print(prof.display())
    return g, d

def train_single_scale(generators, discriminators, opt, dataset):
    
    start_t = time.time()

    torch.manual_seed(0)
    
    # Create the new generator and discriminator for this level
    if(len(generators) == opt['scale_in_training']):
        generator, num_kernels_this_scale = init_gen(len(generators), opt)
        generator = generator.to(opt["device"])
        discriminator = init_discrim(len(generators), opt).to(opt["device"])
    else:
        generator = generators[-1].to(opt['device'])
        generators.pop(len(generators)-1)
        discriminator = discriminators[-1].to(opt['device'])
        discriminators.pop(len(discriminators)-1)

    # Move all models to this GPU and make them distributed
    for i in range(len(generators)):
        generators[i].to(opt["device"])
        generators[i].eval()
        for param in generators[i].parameters():
            param.requires_grad = False

    #print_to_log_and_console(generator, os.path.join(opt["save_folder"], opt["save_name"]),
    #    "log.txt")
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    #print_to_log_and_console("Kernels this scale: %i" % num_kernels_this_scale, 
    #    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[1600-opt['iteration_number']],gamma=opt['gamma'])

    discriminator_optimizer = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
    lr=opt["learning_rate"], betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_optimizer,
    milestones=[1600-opt['iteration_number']],gamma=opt['gamma'])
 
    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))

    start_time = time.time()
    next_save = 0
    volumes_seen = 0

    # Get properly sized frame for this generator
    
    print(str(len(generators)) + ": " + str(opt["resolutions"][len(generators)]))
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=opt["num_workers"]
    )
    for epoch in range(opt['epoch_number'], opt["epochs"]):

        #for iteration in range(len(dataset)):
        for batch_num, real_hr in enumerate(dataloader):
            #print("Original data shape: %s" % str(real_hr.shape))
            if(len(generators) < opt['n'] - 1):
                real_hr = downsample(real_hr, opt['resolutions'][len(generators)], opt['downsample_mode'])
            if(opt['mode'] == '2D' or opt['mode'] == "3Dto2D"):
                if(real_hr.shape[2] > opt['cropping_resolution']):
                    rand_crop_x_start = torch.randint(real_hr.shape[2] - opt['cropping_resolution'], [1])[0]
                    rand_crop_x_end = rand_crop_x_start + opt['cropping_resolution']
                else:
                    rand_crop_x_start = 0
                    rand_crop_x_end = real_hr.shape[2]
                if(real_hr.shape[3] > opt['cropping_resolution']):
                    rand_crop_y_start = torch.randint(real_hr.shape[3] - opt['cropping_resolution'], [1])[0]
                    rand_crop_y_end = rand_crop_y_start + opt['cropping_resolution']
                else:
                    rand_crop_y_start = 0
                    rand_crop_y_end = real_hr.shape[3]
                real_hr = real_hr[:,:,rand_crop_x_start:rand_crop_x_end,rand_crop_y_start:rand_crop_y_end]
                
            else:
                if(real_hr.shape[2] > opt['cropping_resolution']):
                    rand_crop_x_start = torch.randint(real_hr.shape[2] - opt['cropping_resolution'], [1])[0]
                    rand_crop_x_end = rand_crop_x_start + opt['cropping_resolution']
                else:
                    rand_crop_x_start = 0
                    rand_crop_x_end = real_hr.shape[2]
                if(real_hr.shape[3] > opt['cropping_resolution']):    
                    rand_crop_y_start = torch.randint(real_hr.shape[3] - opt['cropping_resolution'], [1])[0]
                    rand_crop_y_end = rand_crop_y_start + opt['cropping_resolution']
                else:
                    rand_crop_y_start = 0
                    rand_crop_y_end = real_hr.shape[3]
                if(real_hr.shape[4] > opt['cropping_resolution']):
                    rand_crop_z_start = torch.randint(real_hr.shape[4] - opt['cropping_resolution'], [1])[0]
                    rand_crop_z_end = rand_crop_z_start + opt['cropping_resolution']
                else:
                    rand_crop_z_start = 0
                    rand_crop_z_end = real_hr.shape[4]    
                real_hr = real_hr[:,:,rand_crop_x_start:rand_crop_x_end,rand_crop_y_start:rand_crop_y_end,rand_crop_z_start:rand_crop_z_end]
            
            #print("Adjusted HR shape: %s" % str(real_hr.shape))
            real_hr = real_hr.to(opt["device"])
            real_lr = F.interpolate(real_hr, scale_factor=opt['spatial_downscale_ratio'],mode=opt['downsample_mode'],align_corners=False)
            #print("LR shapeafter interp downscale: %s" % str(real_lr.shape))
            real_lr = F.interpolate(real_lr, scale_factor=1/opt['spatial_downscale_ratio'],mode=opt['upsample_mode'],align_corners=False)
            #print("LR shape after interp upscale: %s" % str(real_lr.shape))
            D_loss = 0
            G_loss = 0        
            gradient_loss = 0
            rec_loss = 0        
            g = 0
            mags = np.zeros(1)
            angles = np.zeros(1)
            print(generator.learned_scaling_weights)
            # Update discriminator: maximize D(x) + D(G(z))
            if(opt["alpha_2"] > 0.0):            
                for j in range(opt["discriminator_steps"]):
                    discriminator.zero_grad()
                    generator.zero_grad()
                    D_loss = 0

                    # Train with real downscaled to this scale
                    output_real = discriminator(real_hr)
                    discrim_error_real = -output_real.mean()
                    D_loss += discrim_error_real.mean().item()
                    discrim_error_real.backward(retain_graph=True)

                    fake = generator(real_lr)
                    output_fake = discriminator(fake.detach())
                    discrim_error_fake = output_fake.mean()
                    discrim_error_fake.backward(retain_graph=True)
                    D_loss += discrim_error_fake.item()

                    if(opt['regularization'] == "GP"):
                        # Gradient penalty 
                        gradient_penalty = calc_gradient_penalty(discriminator, real_hr, fake, 1, opt['device'])
                        gradient_penalty.backward()
                    elif(opt['regularization'] == "TV"):
                        # Total variance penalty
                        TV_penalty_real = torch.abs(-discrim_error_real-1)
                        TV_penalty_real.backward(retain_graph=True)
                        TV_penalty_fake = torch.abs(discrim_error_fake+1)
                        TV_penalty_fake.backward(retain_graph=True)
                    discriminator_optimizer.step()

            # Update generator: maximize D(G(z))
            for j in range(opt["generator_steps"]):
                generator.zero_grad()
                discriminator.zero_grad()
                G_loss = 0
                gen_err_total = 0
                phys_loss = 0
                path_loss = 0
                loss = nn.L1Loss().to(opt["device"])
                fake = generator(real_lr)
                if(opt["alpha_2"] > 0.0):                    
                    output = discriminator(fake)
                    generator_error = -output.mean()# * opt["alpha_2"]
                    generator_error.backward(retain_graph=True)
                    gen_err_total += generator_error.mean().item()
                    G_loss = output.mean().item()

                if(opt['alpha_1'] > 0.0):
                    rec_loss = loss(fake, real_hr) * opt["alpha_1"]
                    rec_loss.backward(retain_graph=True)
                    gen_err_total += rec_loss.item()
                    rec_loss = rec_loss.detach()

                if(opt['alpha_3'] > 0.0):
                    if(opt["physical_constraints"] == "soft"):
                        if(opt['mode'] == "2D" or opt['mode'] == '3Dto2D'):
                            g_map = TAD(dataset.unscale(fake), opt["device"])            
                            g = g_map.mean()
                        elif(opt['mode'] == "3D"):
                            g_map = TAD3D_CD(dataset.unscale(fake), opt["device"])
                            g = g_map.mean()
                        phys_loss = opt["alpha_3"] * g 
                        phys_loss.backward(retain_graph=True)
                        gen_err_total += phys_loss.item()
                        phys_loss = phys_loss.item()
                if(opt['alpha_4'] > 0.0):                    
                    cs = torch.nn.CosineSimilarity(dim=1).to(opt['device'])
                    mags = torch.abs(torch.norm(fake, dim=1) - torch.norm(real_hr, dim=1))
                    angles = torch.abs(cs(fake, real_hr) - 1) / 2
                    r_loss = opt['alpha_4'] * (mags.mean() + angles.mean()) / 2
                    r_loss.backward(retain_graph=True)
                    gen_err_total += r_loss.item()

                if(opt['alpha_5'] > 0.0):
                    real_gradient = []
                    rec_gradient = []
                    for ax1 in range(real_hr.shape[1]):
                        for ax2 in range(len(real_hr.shape[2:])):
                            if(opt["mode"] == '2D' or opt['mode'] == '3Dto2D'):
                                r_deriv = spatial_derivative2D(real_hr[:,ax1:ax1+1], ax2, opt['device'])
                                rec_deriv = spatial_derivative2D(fake[:,ax1:ax1+1], ax2, opt['device'])
                            elif(opt['mode'] == '3D'):
                                r_deriv = spatial_derivative3D_CD(real_hr[:,ax1:ax1+1], ax2, opt['device'])
                                rec_deriv = spatial_derivative3D_CD(fake[:,ax1:ax1+1], ax2, opt['device'])
                            real_gradient.append(r_deriv)
                            rec_gradient.append(rec_deriv)
                    real_gradient = torch.cat(real_gradient, 1)
                    rec_gradient = torch.cat(rec_gradient, 1)
                    gradient_loss = loss(real_gradient, rec_gradient)
                    gradient_loss_adj = gradient_loss * opt['alpha_5']
                    gradient_loss_adj.backward(retain_graph=True)
                    gen_err_total += gradient_loss_adj.item()

                if(opt["alpha_6"] > 0):
                    if(opt['mode'] == '3D'):
                        if(opt['adaptive_streamlines']):
                            path_loss = adaptive_streamline_loss3D(real_hr, fake, 
                            torch.abs(mags[0] + angles[0]), int(opt['streamline_res']**3), 
                            3, 1, opt['streamline_length'], opt['device'], 
                            periodic=opt['periodic'])* opt['alpha_6']
                        else:
                            path_loss = streamline_loss3D(real_hr, fake, 
                            opt['streamline_res'], opt['streamline_res'], opt['streamline_res'], 
                            1, opt['streamline_length'], opt['device'], 
                            periodic=opt['periodic'] and fake.shape == real.shape) * opt['alpha_6']

                    elif(opt['mode'] == '2D' or opt['mode'] == '3Dto2D'):
                        path_loss = streamline_loss2D(real_hr, fake, 
                        opt['streamline_res'], opt['streamline_res'], 
                        1, opt['streamline_length'], opt['device'], periodic=opt['periodic']) * opt['alpha_6']

                    path_loss.backward(retain_graph=True)
                    path_loss = path_loss.item()
                
                generator_optimizer.step()
            volumes_seen += 1

            if(volumes_seen % 50 == 0):
                rec_numpy = fake.detach().cpu().numpy()[0]
                rec_cm = toImg(rec_numpy)
                rec_cm -= rec_cm.min()
                rec_cm *= (1/rec_cm.max())
                writer.add_image("reconstructed/%i"%len(generators), 
                rec_cm.clip(0,1), volumes_seen)

                real_numpy = real_hr.detach().cpu().numpy()[0]
                real_cm = toImg(real_numpy)
                real_cm -= real_cm.min()
                real_cm *= (1/real_cm.max())
                writer.add_image("real/%i"%len(generators), 
                real_cm.clip(0,1), volumes_seen)

                if(opt["alpha_3"] > 0.0):
                    g_cm = toImg(g_map.detach().cpu().numpy()[0])
                    writer.add_image("Divergence/%i"%len(generators), 
                    g_cm, volumes_seen)

                if(opt["alpha_4"] > 0.0):
                    angles_cm = toImg(angles.detach().cpu().numpy())
                    writer.add_image("angle/%i"%len(generators), 
                    angles_cm , volumes_seen)

                    mags_cm = toImg(mags.detach().cpu().numpy())
                    writer.add_image("mag/%i"%len(generators), 
                    mags_cm, volumes_seen)

            print_to_log_and_console("%i/%i: Dloss=%.02f Gloss=%.02f L1=%.04f AMD=%.02f AAD=%.02f" %
            (volumes_seen, opt['epochs']*len(dataset), D_loss, G_loss, rec_loss, mags.mean(), angles.mean()), 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

            writer.add_scalar('D_loss_scale/%i'%len(generators), D_loss, volumes_seen) 
            writer.add_scalar('G_loss_scale/%i'%len(generators), G_loss, volumes_seen) 
            writer.add_scalar('L1/%i'%len(generators), rec_loss, volumes_seen)
            writer.add_scalar('Gradient_loss/%i'%len(generators), gradient_loss / (opt['alpha_5']+1e-6), volumes_seen)
            writer.add_scalar('TAD/%i'%len(generators), phys_loss / (opt["alpha_3"]+1e-6), volumes_seen)
            writer.add_scalar('path_loss/%i'%len(generators), path_loss / (opt['alpha_6']+1e-6), volumes_seen)
            writer.add_scalar('Mag_loss_scale/%i'%len(generators), mags.mean(), volumes_seen) 
            writer.add_scalar('Angle_loss_scale/%i'%len(generators), angles.mean(), volumes_seen) 

            discriminator_scheduler.step()
            generator_scheduler.step()  

        if(epoch % opt['save_every'] == 0):
            opt["iteration_number"] = batch_num
            opt["epoch_number"] = epoch
            save_models(generators + [generator], discriminators + [discriminator], opt)
        

    generator = reset_grads(generator, False)
    generator.eval()
    discriminator = reset_grads(discriminator, False)
    discriminator.eval()

    return generator, discriminator

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

        if(self.opt['scaling_mode'] == "learned"):
            
            self.learned_scaling_weights = torch.ones([self.opt['num_channels']])
            self.learned_scaling_bias = torch.ones([self.opt['num_channels']]) * 0.001
            self.learned_scaling_weights.requires_grad = True
            self.learned_scaling_bias.requires_grad = True
            

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
        
        if(self.opt['scaling_mode'] == "learned"):
            for i in range(self.opt['num_channels']):
                data[:,i] = data[:,i] * self.learned_scaling_weights[i]
                data[:,i] = data[:,i] + self.learned_scaling_bias[i]
                
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
        elif(self.opt['scaling_mode'] == "learned"):
            output = data + output
            for i in range(self.opt['num_channels']):
                output[:,i] = output[:,i] - self.learned_scaling_bias[i]
                output[:,i] = output[:,i] / self.learned_scaling_weights[i]
            return output
        else:
            return output + data

class Discriminator(nn.Module):
    def __init__ (self, resolution, num_kernels, opt):
        super(Discriminator, self).__init__()
        
        self.resolution = resolution
        self.num_kernels = num_kernels

        use_sn = opt['regularization'] == "SN"
        if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d

        modules = []
        for i in range(opt['num_blocks']):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, opt['num_channels'], num_kernels, 
                    opt['kernel_size'], opt['stride'], 0, use_sn),
                    create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == opt['num_blocks']-1:  
                tail = nn.Sequential(
                    create_conv_layer(conv_layer, num_kernels, 1, 
                    opt['kernel_size'], opt['stride'], 0, use_sn)
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, num_kernels, num_kernels, 
                    opt['kernel_size'], opt['stride'], 0, use_sn),
                    create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model =  nn.Sequential(*modules)
        self.model = self.model.to(opt['device'])

    def receptive_field(self):
        return (self.opt['kernel_size']-1)*self.opt['num_blocks']

    def forward(self, x):
        return self.model(x)

def create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn):
    bnl = batchnorm_layer(num_kernels)
    bnl.apply(weights_init)
    if(use_sn):
        bnl = SpectralNorm(bnl)
    return bnl

def create_conv_layer(conv_layer, in_chan, out_chan, kernel_size, stride, padding, use_sn):
    c = conv_layer(in_chan, out_chan, 
                    kernel_size, stride, 0)
    c.apply(weights_init)
    if(use_sn):
        c = SpectralNorm(c)
    return c

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0

        for filename in os.listdir(self.opt['data_folder']):
            if(self.num_items == 0):
                d = np.load(os.path.join(self.opt['data_folder'], filename))
                print(filename + " " + str(d.shape))                
                self.num_channels = d.shape[0]
                self.resolution = d.shape[1:]
                if(self.opt['mode'] == "3Dto2D"):
                    self.resolution = self.resolution[0:len(self.resolution)-1]

            if(self.num_items > 0 and (opt['scaling_mode'] == "magnitude" or opt['scaling_mode'] == "channel")):
                d = np.load(os.path.join(self.opt['data_folder'], filename))

            if(opt['scaling_mode'] == "magnitude"):  
                mags = np.linalg.norm(d, axis=0)
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