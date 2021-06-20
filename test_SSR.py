from spatial_models import Generator, load_models
from options import Options, load_options
from utility_functions import streamline_loss3D, str2bool, AvgPool3D, AvgPool2D
import numpy as np
import os
import imageio
import argparse
import time
import datetime
from math import log2, log, log10
import pandas as pd
import pickle
from datasets import TestingDataset
import torch
import torch.nn.functional as F
import imageio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
from math import exp
from typing import Dict, List, Tuple, Optional
from utility_functions import ssim, ssim3D, save_obj, load_obj

class img_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]


def mse_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    return ((GT-x)**2).mean()

def psnr_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (20.0*torch.log10(data_range)-10.0*torch.log10(mse_func(GT, x, device)))

def mre_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (torch.abs(GT-x).max() / data_range)

def mag_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    return torch.abs(torch.norm(GT, dim=1) - torch.norm(x, dim=1)).mean()

def angle_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    cs = torch.nn.CosineSimilarity(dim=1).to(device)
    return (torch.abs(cs(GT,x) - 1) / 2).mean()

def streamline_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    vals = []
    for i in range(100):
        vals.append(streamline_loss3D(GT, x,
        100, 100, 100, 
        1, 5, device, False).item())
    vals = np.array(vals)
    return vals.mean(), vals.std()

def energy_spectra_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    print("to be implemented")
    return 0

def volume_to_imgs(volume, device):
    imgs = []

    im = volume[0,:,:,:,:].permute(1, 0, 2, 3)
    im -= volume.min()
    im *= (255/(volume.max()-volume.min()))#.type(torch.uint8)
    imgs.append(im)

    im = volume[0,:,:,:,:].permute(2, 0, 1, 3)
    im -= volume.min()
    im *= (255/(volume.max()-volume.min()))#.type(torch.uint8)
    imgs.append(im)

    im = volume[0,:,:,:,:].permute(3, 0, 1, 2)
    im -= volume.min()
    im *= (255/(volume.max()-volume.min()))#.type(torch.uint8)
    imgs.append(im)

    return torch.cat(imgs, dim=0)
    
def img_psnr_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    m = ((GT-x)**2).mean()
    return (20.0*log(255.0)-10.0*log(m)).item()

def generate_by_patch(generator, input_volume, patch_size, receptive_field, device):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(device)
        
        rf = receptive_field
                    
        z_done = False
        z = 0
        z_stop = min(input_volume.shape[2], z + patch_size)
        while(not z_done):
            if(z_stop == input_volume.shape[2]):
                z_done = True
            y_done = False
            y = 0
            y_stop = min(input_volume.shape[3], y + patch_size)
            while(not y_done):
                if(y_stop == input_volume.shape[3]):
                    y_done = True
                x_done = False
                x = 0
                x_stop = min(input_volume.shape[4], x + patch_size)
                while(not x_done):                        
                    if(x_stop == input_volume.shape[4]):
                        x_done = True
                    print("%d:%d, %d:%d, %d:%d" % (z, z_stop, y, y_stop, x, x_stop))
                    result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])

                    x_offset = rf if x > 0 else 0
                    y_offset = rf if y > 0 else 0
                    z_offset = rf if z > 0 else 0

                    final_volume[:,:,
                    2*z+z_offset:2*z+result.shape[2],
                    2*y+y_offset:2*y+result.shape[3],
                    2*x+x_offset:2*x+result.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                    x += patch_size - 2*rf
                    x = min(x, max(0, input_volume.shape[4] - patch_size))
                    x_stop = min(input_volume.shape[4], x + patch_size)
                y += patch_size - 2*rf
                y = min(y, max(0, input_volume.shape[3] - patch_size))
                y_stop = min(input_volume.shape[3], y + patch_size)
            z += patch_size - 2*rf
            z = min(z, max(0, input_volume.shape[2] - patch_size))
            z_stop = min(input_volume.shape[2], z + patch_size)

    return final_volume

def generate_patch(z,z_stop,y,y_stop,x,x_stop,available_gpus):

    device = None
    while(device is None):        
        device, generator, input_volume = available_gpus.get_next_available()
        time.sleep(1)
    #print("Starting SR on device " + device)
    with torch.no_grad():
        result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])
    return result,z,z_stop,y,y_stop,x,x_stop,device

class SharedList(object):  
    def __init__(self, items, generators, input_volumes):
        self.lock = threading.Lock()
        self.list = items
        self.generators = generators
        self.input_volumes = input_volumes
        
    def get_next_available(self):
        #print("Waiting for a lock")
        self.lock.acquire()
        item = None
        generator = None
        input_volume = None
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            if(len(self.list) > 0):                    
                item = self.list.pop(0)
                generator = self.generators[item]
                input_volume = self.input_volumes[item]
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()
        return item, generator, input_volume
    
    def add(self, item):
        #print("Waiting for a lock")
        self.lock.acquire()
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            self.list.append(item)
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()

def generate_by_patch_parallel(generator, input_volume, patch_size, receptive_field, devices):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(devices[0])
        
        rf = receptive_field

        available_gpus = []
        generators = {}
        input_volumes = {}

        for i in range(1, len(devices)):
            available_gpus.append(devices[i])
            g = copy.deepcopy(generator).to(devices[i])
            iv = input_volume.clone().to(devices[i])
            generators[devices[i]] = g
            input_volumes[devices[i]] = iv
            torch.cuda.empty_cache()

        available_gpus = SharedList(available_gpus, generators, input_volumes)

        threads= []
        with ThreadPoolExecutor(max_workers=len(devices)-1) as executor:
            z_done = False
            z = 0
            z_stop = min(input_volume.shape[2], z + patch_size)
            while(not z_done):
                if(z_stop == input_volume.shape[2]):
                    z_done = True
                y_done = False
                y = 0
                y_stop = min(input_volume.shape[3], y + patch_size)
                while(not y_done):
                    if(y_stop == input_volume.shape[3]):
                        y_done = True
                    x_done = False
                    x = 0
                    x_stop = min(input_volume.shape[4], x + patch_size)
                    while(not x_done):                        
                        if(x_stop == input_volume.shape[4]):
                            x_done = True
                        
                        
                        threads.append(
                            executor.submit(
                                generate_patch,
                                z,z_stop,
                                y,y_stop,
                                x,x_stop,
                                available_gpus
                            )
                        )
                        
                        x += patch_size - 2*rf
                        x = min(x, max(0, input_volume.shape[4] - patch_size))
                        x_stop = min(input_volume.shape[4], x + patch_size)
                    y += patch_size - 2*rf
                    y = min(y, max(0, input_volume.shape[3] - patch_size))
                    y_stop = min(input_volume.shape[3], y + patch_size)
                z += patch_size - 2*rf
                z = min(z, max(0, input_volume.shape[2] - patch_size))
                z_stop = min(input_volume.shape[2], z + patch_size)

            for task in as_completed(threads):
                result,z,z_stop,y,y_stop,x,x_stop,device = task.result()
                result = result.to(devices[0])
                x_offset_start = rf if x > 0 else 0
                y_offset_start = rf if y > 0 else 0
                z_offset_start = rf if z > 0 else 0
                x_offset_end = rf if x_stop < input_volume.shape[4] else 0
                y_offset_end = rf if y_stop < input_volume.shape[3] else 0
                z_offset_end = rf if z_stop < input_volume.shape[2] else 0
                #print("%d, %d, %d" % (z, y, x))
                final_volume[:,:,
                2*z+z_offset_start:2*z+result.shape[2] - z_offset_end,
                2*y+y_offset_start:2*y+result.shape[3] - y_offset_end,
                2*x+x_offset_start:2*x+result.shape[4] - x_offset_end] = result[:,:,
                z_offset_start:result.shape[2]-z_offset_end,
                y_offset_start:result.shape[3]-y_offset_end,
                x_offset_start:result.shape[4]-x_offset_end]
                available_gpus.add(device)
    
    return final_volume

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--mode',default="3D",type=str,help='2D or 3D')
    parser.add_argument('--testing_method',default="model",type=str,help='What method to test, model or trilinear')
    parser.add_argument('--scale_factor',default=2,type=int,help='2x, 4x, 8x... what the model supports')
    parser.add_argument('--full_resolution',default=1024,type=int,help='The full resolution of the frame')
    parser.add_argument('--channels',default=3,type=int,help='Number of channels in the data')
    parser.add_argument('--data_folder',default="iso1024",type=str,help='Name of folder with test data in /TestingData')
    parser.add_argument('--model_name',default="SSR",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cpu",type=str,help='Device to use for testing')
    parser.add_argument('--parallel',default="False",type=str2bool,help='Perform SR in parallel')
    parser.add_argument('--print',default="True",type=str2bool,help='Print output during testing')
    parser.add_argument('--debug',default="False",type=str2bool,help='Use fake data during testing instead of loading')
    parser.add_argument('--test_on_gpu',default="True",type=str2bool,help='Metrics calculated on GPU?')
    parser.add_argument('--fix_dim_order',default="False",type=str2bool,help='True if channels are last')

    parser.add_argument('--test_mse',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_psnr',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_amd',default="False",type=str2bool,help='Enables tests for average magnitude difference')
    parser.add_argument('--test_aad',default="False",type=str2bool,help='Enables tests for average angle difference')
    parser.add_argument('--test_mre',default="True",type=str2bool,help='Enables tests for maximum relative error')
    parser.add_argument('--test_ssim',default="True",type=str2bool,help='Enables tests for maximum relative error')
    parser.add_argument('--test_streamline',default="False",type=str2bool,help='Enables streamline error tests')

    parser.add_argument('--test_img_psnr',default="False",type=str2bool,help='Enables tests for image PSNR score')
    parser.add_argument('--test_img_ssim',default="False",type=str2bool,help='Enables tests for image SSIM score')
    parser.add_argument('--test_img_fid',default="False",type=str2bool,help='Enables tests for image FID score')

    parser.add_argument('--save_name',default="SSR",type=str,help='Where to write results')
    parser.add_argument('--output_file_name',default="SSR.pkl",type=str,help='Where to write results')
    
    args = vars(parser.parse_args())

    p = args['print']

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "TestingData", args['data_folder'])
    save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    if(p):
        print("Loading options and model")
        print("full resolution: " + str(args['full_resolution']))
        print("channels: " + str(args['channels']))
    opt = load_options(os.path.join(save_folder, args["model_name"]))
    opt['cropping_resolution'] = args['full_resolution']
    opt["device"] = args["device"]
    opt['data_folder'] = "TestingData/"+args['data_folder']
    generators, _ = load_models(opt,"cpu")
    for i in range(len(generators)):
        generators[i] = generators[i].to(opt['device'])
        generators[i].train(False)

    dataset = TestingDataset(opt['data_folder'])
    results_location = os.path.join(output_folder, args['output_file_name'])

    if(torch.cuda.device_count() > 1 and args['parallel']):
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append("cuda:"+str(i))

    d = {
        "inference_time": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
        "mre": [],
        "streamline_error_mean": [],
        "streamline_error_std": [],
        "inner_psnr": [],
        "inner_mre": [],
        "img_psnr": [],
        "img_ssim": [],
        "img_fid": [],
        "aad": [],
        "amd": []
    }

    images = []

    with torch.no_grad():
        for i in range(len(dataset)):
            if(p):
                print("Loading dataset item : " + str(i) + " name: " + dataset.item_names[i])
            start_load_time = time.time()
            if(args['debug']):
                if(args['mode'] == "3D"):
                    GT_data = torch.randn([1, args['channels'], args['full_resolution'], args['full_resolution'], args['full_resolution']]).to(args['device'])
                elif(args['mode'] == "2D"):
                    GT_data = torch.randn([1, args['channels'], args['full_resolution'], args['full_resolution']]).to(args['device'])
            else:
                GT_data = dataset[i].to(args['device'])
            if(args['fix_dim_order']):
                GT_data = GT_data.permute(0, 4, 1, 2, 3)
            end_load_time = time.time()
            GT_data.requires_grad_(False)
            if(p):
                print("Data size: " + str(GT_data.shape))
                print("Finished loading in " + str(end_load_time-start_load_time) + \
                    ". Downscaling by " + str(args['scale_factor']))

            
            if(opt['downsample_mode'] == "average_pooling"):
                chans = []
                if(args['mode'] == "3D"):
                    for j in range(args['channels']):
                        LR_data = AvgPool3D(GT_data[:,j:j+1,:,:,:], args['scale_factor'])
                        chans.append(LR_data)
                elif(args['mode'] == "2D"):
                    for j in range(args['channels']):
                        LR_data = AvgPool2D(GT_data[:,j:j+1,:,:], args['scale_factor'])
                        chans.append(LR_data)
                LR_data = torch.cat(chans, dim=1)
            elif(opt['downsample_mode'] == "subsampling"):
                if(args['mode'] == "3D"):
                    LR_data = GT_data[:,:,::args['scale_factor'], ::args['scale_factor'],::args['scale_factor']].clone()
                elif(args['mode'] == "2D"):
                    LR_data = GT_data[:,:,::args['scale_factor'], ::args['scale_factor']].clone()

            if opt['scaling_mode'] == "channel" and args['testing_method'] == "model":
                mins = []
                maxs = []
                for c in range(GT_data.shape[1]):
                    mins.append(GT_data[:,c].min())
                    maxs.append(GT_data[:,c].max())
                    LR_data[:,c] -= mins[-1]
                    LR_data[:,c] *= (1/(maxs[-1]-mins[-1]))
            if(not args['test_on_gpu']):
                GT_data = GT_data.to("cpu")
            torch.cuda.empty_cache()
            if(p):
                print("Finished downscaling to " + str(LR_data.shape) + ". Performing super resolution")
            
            inference_start_time = time.time()
            if(args['testing_method'] == "model"):
                current_ds = args['scale_factor']
                while(current_ds > 1):
                    gen_to_use = int(len(generators) - log2(current_ds))
                    if(torch.cuda.device_count() > 1 and args['parallel'] and args['mode'] == '3D'):
                        if(p):
                            print("Upscaling in parallel on " + str(len(devices)) + " gpus")
                        LR_data = generate_by_patch_parallel(generators[gen_to_use], 
                        LR_data, 140, 10, devices)
                    else:
                        if(args['mode'] == '3D'):
                            LR_data = generate_by_patch(generators[gen_to_use], 
                            LR_data, 140, 10, args['device'])
                        elif(args['mode'] == '2D'):
                            with torch.no_grad():
                                LR_data = generators[gen_to_use](LR_data)
                    current_ds = int(current_ds / 2)
            else:
                if(args['mode'] == "3D"):
                    LR_data = F.interpolate(LR_data, scale_factor=args['scale_factor'], 
                    mode="trilinear", align_corners=True)
                elif(args['mode'] == '2D'):
                    LR_data = F.interpolate(LR_data, scale_factor=args['scale_factor'], 
                    mode=args['testing_method'], align_corners=True)
            inference_end_time = time.time()
            
            inference_this_frame = inference_end_time - inference_start_time
            if opt['scaling_mode'] == "channel" and args['testing_method'] == "model":
                for c in range(GT_data.shape[1]):
                    LR_data[:,c] *= (maxs[c]-mins[c])
                    LR_data[:,c] += mins[c]
            if(p):
                print("Finished super resolving in %0.04f seconds. Final shape: %s. Performing tests." % \
                    (inference_this_frame, str(LR_data.shape)))
            if(not args['test_on_gpu']):
                LR_data = LR_data.to("cpu")

            mse_this_frame = None
            psnr_this_frame = None
            mre_this_frame = None
            img_psnr_this_frame = None
            img_ssim_this_frame = None
            img_fid_this_frame = None

            d['inference_time'].append(inference_this_frame)

            if(args['mode'] == '3D'):
                LR_img_this_frame = LR_data[0,:,int(LR_data.shape[2]/2),:,:].clone().cpu()
                GT_img_this_frame = GT_data[0,:,int(GT_data.shape[2]/2),:,:].clone().cpu()
            elif(args['mode'] == '2D'):
                LR_img_this_frame = LR_data[0,:,:,:].clone().cpu()
                GT_img_this_frame = GT_data[0,:,:,:].clone().cpu()

            LR_img_this_frame -= GT_data.min().cpu()
            GT_img_this_frame -= GT_data.min().cpu()

            LR_img_this_frame *= (255/(GT_data.max().cpu()-GT_data.min().cpu()))
            GT_img_this_frame *= (255/(GT_data.max().cpu()-GT_data.min().cpu()))

            LR_img_this_frame = LR_img_this_frame.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            GT_img_this_frame = GT_img_this_frame.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

            img_folder = os.path.join(output_folder, str(args['scale_factor']) + \
                "x_imgs")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            
            LR_img_name = os.path.join(img_folder, dataset.item_names[i]+\
                "_"+args['save_name']+".png")
            GT_img_name = os.path.join(img_folder, dataset.item_names[i]+\
                "_GT.png")
            imageio.imwrite(LR_img_name, LR_img_this_frame)
            imageio.imwrite(GT_img_name, GT_img_this_frame)
            

            if(args['test_mse']):
                mse_item = mse_func(GT_data, LR_data, args['device'] if args['test_on_gpu'] else "cpu").item()
                if(p):
                    print("MSE: " + str(mse_item))
                d['mse'].append(mse_item)

            if(args['test_psnr']):
                psnr_item = psnr_func(
                    GT_data,
                    LR_data, 
                    args['device'] if args['test_on_gpu'] else "cpu").item()
                if(p):
                    print("PSNR: " + str(psnr_item))
                d['psnr'].append(psnr_item)

                if(args['mode'] == '2D'):
                    inner_psnr_item = psnr_func(
                        GT_data[:,:,6:GT_data.shape[2]-6,
                                6:GT_data.shape[3]-6], 
                                LR_data[:,:,6:LR_data.shape[2]-6,
                                6:LR_data.shape[3]-6], args['device'] if args['test_on_gpu'] else "cpu").item()
                elif(args['mode'] == '3D'):
                    inner_psnr_item = psnr_func(
                        GT_data[:,:,6:GT_data.shape[2]-6,
                                6:GT_data.shape[3]-6,
                                6:GT_data.shape[4]-6], 
                                LR_data[:,:,6:LR_data.shape[2]-6,
                                6:LR_data.shape[3]-6,
                                6:LR_data.shape[4]-6], args['device'] if args['test_on_gpu'] else "cpu").item()
                if(p):
                    print("Inner PSNR: " + str(inner_psnr_item))
                d['inner_psnr'].append(inner_psnr_item)

            if(args['test_mre']):
                mre_item = mre_func(GT_data, LR_data, args['device'] if args['test_on_gpu'] else "cpu").item()
                if(p):
                    print("MRE: " + str(mre_item))
                d['mre'].append(mre_item)
            
            if(args['test_mre']):
                if(args['mode'] == '2D'):
                    mre_item = mre_func(
                        GT_data[:,:,6:GT_data.shape[2]-6,
                                6:GT_data.shape[3]-6], 
                                LR_data[:,:,6:LR_data.shape[2]-6,
                                6:LR_data.shape[3]-6], args['device'] if args['test_on_gpu'] else "cpu").item()
                elif(args['mode'] == '3D'):
                    mre_item = mre_func(
                        GT_data[:,:,6:GT_data.shape[2]-6,
                                6:GT_data.shape[3]-6,
                                6:GT_data.shape[4]-6], 
                                LR_data[:,:,6:LR_data.shape[2]-6,
                                6:LR_data.shape[3]-6,
                                6:LR_data.shape[4]-6], args['device'] if args['test_on_gpu'] else "cpu").item()
                if(p):
                    print("Inner MRE: " + str(mre_item))
                d['inner_mre'].append(mre_item)
            if(args['test_ssim']):
                if(args['mode'] == '2D'):
                    ssim_item = ssim(GT_data, LR_data).item()
                elif(args['mode'] == '3D'):
                    ssim_item = ssim3D(GT_data, LR_data).item()
                d['ssim'].append(ssim_item)
            if(args['test_streamline']):
                streamline_avg, streamline_std = streamline_func(GT_data, LR_data, args['device'])
                d["streamline_error_mean"].append(streamline_avg)
                d["streamline_error_std"].append(streamline_std)
            
                        
            if(args['test_aad']):
                cs = torch.nn.CosineSimilarity(dim=1).to(args['device'])
                angles = (torch.abs(cs(LR_data[:,:,6:LR_data.shape[2]-6,
                    6:LR_data.shape[3]-6,6:LR_data.shape[4]-6], 
                    GT_data[:,:,6:GT_data.shape[2]-6,
                    6:GT_data.shape[3]-6,6:GT_data.shape[4]-6]) - 1) / 2).mean().item()
                d['aad'].append(angles)

            if(args['test_amd']):
                mags = torch.abs(torch.norm(LR_data[:,:,6:LR_data.shape[2]-6,
                    6:LR_data.shape[3]-6,6:LR_data.shape[4]-6], dim=1) \
                    - torch.norm(GT_data[:,:,6:GT_data.shape[2]-6,
                    6:GT_data.shape[3]-6,6:GT_data.shape[4]-6], dim=1)).mean().item()
                d['amd'].append(mags)

            '''
            if(args['test_img_psnr']):
                psnr_item = img_psnr_func(GT_data, LR_data, "cpu")
                if(p):
                    print("Image PSNR: " + str(psnr_item))
                d['img_psnr'].append(psnr_item)
            '''
    
    if(os.path.exists(results_location)):
        all_data = load_obj(results_location)
        if(p):
            print("Found existing results, will append new results")
    else:
        all_data = {}
    
    all_data[args['save_name']] = d

    save_obj(all_data, results_location)
    
    if(p):
        print("Saved results")
