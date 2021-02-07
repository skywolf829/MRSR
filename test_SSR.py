from spatial_models import *
from options import *
from utility_functions import *
import numpy as np
import os
import imageio
import argparse
import time
import datetime
from pytorch_memlab import LineProfiler, MemReporter, profile
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--testing_method',default="model",type=str,help='What method to test, model or trilinear')
    parser.add_argument('--SR_factor',default=2,type=int,help='2x, 4x, 8x... what the model supports')
    parser.add_argument('--data_folder',default="isotropic1024",type=str,help='Name of folder with test data in /TestingData')
    parser.add_argument('--model_name',default="SSR",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cuda",type=str,help='Device to use for testing')
    parser.add_argument('--print',default="True",type=str2bool,help='Print output during testing')
    parser.add_argument('--test_mse',default="True",type=str2bool,help='Enables tests for mse')
    parser.add_argument('--test_mre',default="True",type=str2bool,help='Enables tests for maximum relative error')
    parser.add_argument('--test_mag',default="True",type=str2bool,help='Enables tests for average magnitude difference')
    parser.add_argument('--test_energy_spectra',default="True",type=str2bool,help='Enables tests for energy spectra')
    parser.add_argument('--test_angle',default="True",type=str2bool,help='Enables tests for average angle difference')
    parser.add_argument('--test_img_psnr',default="True",type=str2bool,help='Enables tests for image PSNR score')
    parser.add_argument('--test_img_ssim',default="True",type=str2bool,help='Enables tests for image SSIM score')
    parser.add_argument('--test_img_fid',default="True",type=str2bool,help='Enables tests for image FID score')

    parser.add_argument('--output_file_name',default="SSR.csv",type=str,help='Where to write results')
    
    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "TestingData", args['data_folder'])
    save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    
    opt = load_options(os.path.join(save_folder, args["model_name"]))
    opt["device"] = args["device"]
    opt['data_folder'] = "TestingData/"+args['data_folder']
    generators, discriminators = load_models(opt,args["device"])
    dataset = LocalDataset(opt)


    results = 
