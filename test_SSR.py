from spatial_models import *
from options import *
from utility_functions import *
import numpy as np
import os
import imageio
import argparse
import time
import datetime
from math import log2, log
from pytorch_memlab import LineProfiler, MemReporter, profile
import pandas as pd

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def mse_func(GT, x, device):
    return ((GT-x)**2).mean().item()

def psnr_func(GT, x, device):
    data_range = GT.max() - GT.min()
    return (20*log(data_range)-10*log(mse_func(GT, x))).item()

def mre_func(GT, x, device):
    data_range = GT.max() - GT.min()
    return (torch.abs(GT-x).max() / data_range).item()

def mag_func(GT, x, device):
    return torch.abs(torch.norm(GT, dim=1) - torch.norm(x, dim=1)).mean().item()

def angle_func(GT, x, device):
    cs = torch.nn.CosineSimilarity(dim=1).to(device)
    return (torch.abs(cs(fake,real_hr) - 1) / 2).item()

def energy_spectra_func(GT, x, device):
    print("to be implemented")

def volume_to_imgs(volume, device):
    

def img_psnr_func(GT, x, device):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--testing_method',default="model",type=str,help='What method to test, model or trilinear')
    parser.add_argument('--SR_factor',default=2,type=int,help='2x, 4x, 8x... what the model supports')
    parser.add_argument('--data_folder',default="isotropic1024",type=str,help='Name of folder with test data in /TestingData')
    parser.add_argument('--model_name',default="SSR",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cuda",type=str,help='Device to use for testing')
    parser.add_argument('--print',default="True",type=str2bool,help='Print output during testing')

    parser.add_argument('--test_mse',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_psnr',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_mre',default="True",type=str2bool,help='Enables tests for maximum relative error')
    parser.add_argument('--test_mag',default="True",type=str2bool,help='Enables tests for average magnitude difference')
    parser.add_argument('--test_angle',default="True",type=str2bool,help='Enables tests for average angle difference')

    parser.add_argument('--test_energy_spectra',default="True",type=str2bool,help='Enables tests for energy spectra')

    parser.add_argument('--test_img_psnr',default="True",type=str2bool,help='Enables tests for image PSNR score')
    parser.add_argument('--test_img_ssim',default="True",type=str2bool,help='Enables tests for image SSIM score')
    parser.add_argument('--test_img_fid',default="True",type=str2bool,help='Enables tests for image FID score')

    parser.add_argument('--output_file_name',default="SSR.csv",type=str,help='Where to write results')
    
    args = vars(parser.parse_args())

    p = args['print']

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "TestingData", args['data_folder'])
    save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    
    opt = load_options(os.path.join(save_folder, args["model_name"]))
    opt["device"] = args["device"]
    opt['data_folder'] = "TestingData/"+args['data_folder']
    generators, _ = load_models(opt,"cpu")
    for i in len(generators):
        generators[i] = generators[i].to(opt['device'])
        generators[i].train(False)

    dataset = LocalDataset(opt)
    results_location = os.path.join(output_folder, args['output_file_name'])


    d = {
        "mse": [],
        "psnr" [],
        "mre": [],
        "mag": [],
        "angle": [],
        "streamline": [],
        "energy_spectra": [],
        "img_psnr": [],
        "img_ssim": [],
        "img_fid": []
    }

    with torch.no_grad():
        for i in range(len(dataset)):
            if(p):
                print("Loading dataset item : " + str(i))
            GT_data = dataset[i]
            if(p):
                print("Finished loading. Downscaling by " + str(opt['scale_factor']))

            
            if(opt['downsample_mode'] == "average_pooling"):
                LR_data = AvgPool3D(GT_data.clone(), opt['scale_factor'])
            elif(opt['downsample_mode'] == "subsampling"):
                LR_data = GT_data[:,:,::opt['scale_factor'], ::opt['scale_factor']].clone()

            if(p):
                print("Finished downscaling. Performing super resolution")
            
            if(opt['testing_method'] == "model"):
                current_ds = opt['scale_factor']
                while(current_ds > 1):
                    gen_to_use = len(generators) - log2(current_ds)
                    LR_data = generators[gen_to_use](LR_data)
                    current_ds = int(current_ds / 2)
            else:
                LR_data = F.interpolate(LR_data, scale_factor=opt['scale_factor'], 
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

            if(opt['test_mse']):
                mse_item = ((GT_data - LR_data)**2).mean().item()
                if(p):
                    print("MSE: " + str(mse_item))
                d['mse'].append(mse_item)
            if(opt['test_psnr']):

