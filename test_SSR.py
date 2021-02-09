from spatial_models import Generator, load_models
from options import Options, load_options
from utility_functions import streamline_loss3D, str2bool, AvgPool3D
import numpy as np
import os
import imageio
import argparse
import time
import datetime
from math import log2, log
import pandas as pd
import pickle
from datasets import TestingDataset
import torch
import torch.nn.functional as F

class img_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]

def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

def mse_func(GT, x, device):
    return ((GT-x)**2).mean().item()

def psnr_func(GT, x, device):
    data_range = GT.max() - GT.min()
    return (20.0*log(data_range)-10.0*log(mse_func(GT, x, device))).item()

def mre_func(GT, x, device):
    data_range = GT.max() - GT.min()
    return (torch.abs(GT-x).max() / data_range).item()

def mag_func(GT, x, device):
    return torch.abs(torch.norm(GT, dim=1) - torch.norm(x, dim=1)).mean().item()

def angle_func(GT, x, device):
    cs = torch.nn.CosineSimilarity(dim=1).to(device)
    return (torch.abs(cs(GT,x) - 1) / 2).item()

def streamline_func(GT, x, device):
    vals = []
    for i in range(100):
        vals.append(streamline_loss3D(GT, x,
        100, 100, 100, 
        1, 5, device, True).item())
    vals = np.array(vals)
    return vals.mean(), vals.std()

def energy_spectra_func(GT, x, device):
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
    m = ((GT-x)**2).mean()
    return (20.0*log(255.0)-10.0*log(m)).item()

'''
def img_ssim_func(GT, x, device):
    return ssim(x, GT, data_range=1.0).item()

def img_fid_func(GT, x, device):
    GT_dl = torch.utils.data.DataLoader(img_dataset(GT))
    x_dl = torch.utils.data.DataLoader(img_dataset(x))
    fid_metric = FID()
    GT_feats = fid_metric.compute_feats(GT_dl)
    x_feats = fid_metric.compute_feats(x_dl)
    return fid_metric(GT_feats, x_feats)
'''

def generate_by_patch(generator, input_volume, patch_size, receptive_field, device):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(device)
        
        #print("Gen " + str(i))
        rf = receptive_field
                    
        z_done = False
        z = 0
        z_stop = min(final_volume.shape[2], z + patch_size)
        while(not z_done):
            if(z_stop == final_volume.shape[2]):
                z_done = True
            y_done = False
            y = 0
            y_stop = min(final_volume.shape[3], y + patch_size)
            while(not y_done):
                if(y_stop == final_volume.shape[3]):
                    y_done = True
                x_done = False
                x = 0
                x_stop = min(final_volume.shape[4], x + patch_size)
                while(not x_done):                        
                    if(x_stop == final_volume.shape[4]):
                        x_done = True

                    result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])

                    x_offset = rf if x > 0 else 0
                    y_offset = rf if y > 0 else 0
                    z_offset = rf if z > 0 else 0

                    final_volume[:,:,
                    z+z_offset:z+result.shape[2],
                    y+y_offset:y+result.shape[3]],
                    x+x_offset:x+result.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                    x += patch_size - 2*rf
                    x = min(x, max(0, final_volume.shape[4] - patch_size))
                    x_stop = min(final_volume.shape[4], x + patch_size)
                y += patch_size - 2*rf
                y = min(y, max(0, final_volume.shape[3] - patch_size))
                y_stop = min(final_volume.shape[3], y + patch_size)
            z += patch_size - 2*rf
            z = min(z, max(0, final_volume.shape[2] - patch_size))
            z_stop = min(final_volume.shape[2], z + patch_size)

    return final_volume

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--testing_method',default="model",type=str,help='What method to test, model or trilinear')
    parser.add_argument('--scale_factor',default=2,type=int,help='2x, 4x, 8x... what the model supports')
    parser.add_argument('--full_resolution',default=1024,type=int,help='The full resolution of the frame')
    parser.add_argument('--data_folder',default="iso1024",type=str,help='Name of folder with test data in /TestingData')
    parser.add_argument('--model_name',default="SSR",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cpu",type=str,help='Device to use for testing')
    parser.add_argument('--print',default="True",type=str2bool,help='Print output during testing')

    parser.add_argument('--test_mse',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_psnr',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_mre',default="True",type=str2bool,help='Enables tests for maximum relative error')
    parser.add_argument('--test_mag',default="True",type=str2bool,help='Enables tests for average magnitude difference')
    parser.add_argument('--test_angle',default="True",type=str2bool,help='Enables tests for average angle difference')
    parser.add_argument('--test_streamline',default="True",type=str2bool,help='Enables tests for streamline differences')

    parser.add_argument('--test_energy_spectra',default="True",type=str2bool,help='Enables tests for energy spectra')

    parser.add_argument('--test_img_psnr',default="True",type=str2bool,help='Enables tests for image PSNR score')
    parser.add_argument('--test_img_ssim',default="True",type=str2bool,help='Enables tests for image SSIM score')
    parser.add_argument('--test_img_fid',default="True",type=str2bool,help='Enables tests for image FID score')

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


    d = {
        "mse": [],
        "psnr": [],
        "mre": [],
        "mag": [],
        "angle": [],
        "streamline_average": [],
        "streamline_std": [],
        "energy_spectra": [],
        "img_psnr": [],
        "img_ssim": [],
        "img_fid": []
    }

    with torch.no_grad():
        for i in range(len(dataset)):
            if(p):
                print("Loading dataset item : " + str(i))
            GT_data = dataset[i].to(opt['device'])
            if(p):
                print("Data size: " + str(GT_data.shape))
                print("Finished loading. Downscaling by " + str(args['scale_factor']))

            
            if(opt['downsample_mode'] == "average_pooling"):
                LR_data = AvgPool3D(GT_data.clone(), args['scale_factor'])
            elif(opt['downsample_mode'] == "subsampling"):
                LR_data = GT_data[:,:,::args['scale_factor'], ::args['scale_factor']].clone()

            if(p):
                print("Finished downscaling. Performing super resolution")
            
            if(args['testing_method'] == "model"):
                current_ds = args['scale_factor']
                while(current_ds > 1):
                    gen_to_use = int(len(generators) - log2(current_ds))
                    LR_data = generate_by_patch(generators[gen_to_use], LR_data, 96, 10, args['device'])
                    #LR_data = generators[gen_to_use](LR_data)
                    current_ds = int(current_ds / 2)
            else:
                LR_data = F.interpolate(LR_data, scale_factor=args['scale_factor'], 
                mode="trilinear", align_corners=True)


            if(p):
                print("Finished super resolving. Performing tests.")

            mse_this_frame = None
            psnr_this_frame = None
            mre_this_frame = None
            mag_this_frame = None
            angle_this_frame = None
            energy_spectra_this_frame = None
            img_psnr_this_frame = None
            img_ssim_this_frame = None
            img_fid_this_frame = None

            if(args['test_mse']):
                mse_item = mse_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("MSE: " + str(mse_item))
                d['mse'].append(mse_item)

            if(args['test_psnr']):
                psnr_item = psnr_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("PSNR: " + str(psnr_item))
                d['psnr'].append(mse_item)

            if(args['test_mre']):
                mre_item = mre_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("MRE: " + str(mre_item))
                d['mre'].append(mse_item)

            if(args['test_mag']):
                mag_item = mag_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("Mag: " + str(mag_item))
                d['mag'].append(mag_item)

            if(args['test_angle']):
                angle_item = angle_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("Angle: " + str(angle_item))
                d['angle'].append(angle_item)

            if(args['test_angle']):
                angle_item = angle_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("Angle: " + str(angle_item))
                d['angle'].append(angle_item)

            
            if(args['test_streamline']):
                sl_avg, sl_std = streamline_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("Streamline average/std: " + str(sl_avg) + "/" + str(sl_std))
                d['streamline_average'].append(sl_avg)
                d['streamline_std '].append(sl_std)
            
            if(args['test_img_psnr']):
                psnr_item = img_psnr_func(GT_data, LR_data, args['deivce'])
                if(p):
                    print("Image PSNR: " + str(psnr_item))
                d['img_psnr'].append(psnr_item)
    
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
