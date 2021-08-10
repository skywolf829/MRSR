import os
import numpy as np
import imageio
import argparse
import h5py
import time
import torch
from utility_functions import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--folder',default="octree_files",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')    
    parser.add_argument('--channels',default=1,type=int,help='# channels')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--output_folder',default="mag2D_4010",type=str,help='Where to save results')
    parser.add_argument('--start_value',default=10,type=float,help='PSNR to start tests at')
    parser.add_argument('--end_value',default=100,type=float,help='PSNR to end tests at')
    parser.add_argument('--value_skip',default=10,type=float,help='PSNR increment by')
    parser.add_argument('--metric',default='psnr',type=str)
    parser.add_argument('--save_netcdf',default="false",type=str2bool)
    parser.add_argument('--save_TKE',default="false",type=str2bool)
    parser.add_argument('--device',default="cpu",type=str)
    

    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "TestingData", args['folder'])
    
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['output_folder'])

    if(not os.path.exists(save_folder)):
        os.makedirs(save_folder)
    
    # psnr -> metrics
    results = {}
    results['file_size'] = []
    results['psnrs'] = []
    results['rec_psnr'] = []
    results['rec_ssim'] = []
    results['compression_time'] = []
    results['rec_mre'] = []
    results['rec_pwmre'] = []
    results['rec_inner_mre'] = []
    results['rec_inner_pwmre'] = []
    if(args['save_TKE']):
        results['TKE_error'] = []
    
    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
        if("TTHRESH" in all_data.keys()):
            results = all_data['TTHRESH']
        else:
            all_data['TTHRESH'] = results
    else:
        all_data = {}
        all_data['TTHRESH'] = results

    f = h5py.File(os.path.join(input_folder, args['file']), "r")
    d = np.array(f['data'])
    f.close()
    for i in range(args['channels']):
        d[i].tofile(args['file'] + ".dat")

    value = args['start_value']
    while(value < args['end_value']):
        data_channels = []
        f_size_kb = 0
        for i in range(args['channels']):            
            command = "tthresh -i " + args['file'] + ".dat -s" + \
                str(args['nx']) + " " + str(args['ny'])
            if(args['dims'] == 3):
                command = command + " " + str(args['nz'])
            if(args['metric'] == "psnr"):
                command = command + " -p " + str(value)
            elif(args['metric'] == "mre"):
                command = command + " -e " + str(value)
            command = command + " -c " + args['file'] + ".dat.tthresh"
            start_t = time.time()
            print("Running: " + command)
            os.system(command)
            compression_time = time.time() - start_t

            f_size_kb += os.path.getsize(args['file'] + ".dat.tthresh") / 1024

            command = "tthresh -c " + args['file'] + ".dat.tthresh -o " + args['file'] + ".dat.tthresh.out"  

            os.system(command)

            dc = np.fromfile(args['file']+".dat.tthresh.out")
            dc.dtype = np.float32
            if(args['dims'] == 2):
                dc = dc.reshape(args['nx'], args['ny'])
            elif(args['dims'] == 3):
                dc = dc.reshape(args['nx'], args['ny'], args['nz'])    
            data_channels.append(dc)
            command = "mv " + args['file']+".dat.tthresh " + save_folder + \
                "/psnr_"+str(value)+"_"+args['file']+"_"+str(i)+".tthresh"
            os.system(command)
        dc = np.stack(data_channels)

        rec_psnr = PSNR(dc, d)
        final_mre : float = relative_error(dc, d).item()
        final_pwmre: float = pw_relative_error(dc, d).item()
        #rec_ssim = ssim(d, dc)
                
        if(args['dims'] == 2):
            rec_ssim = ssim(torch.Tensor(dc).unsqueeze(0).to(args['device']), 
                torch.Tensor(d).unsqueeze(0).to(args['device'])).cpu().item()
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
        elif(args['dims'] == 3):      
            rec_ssim = ssim3D(torch.Tensor(dc).unsqueeze(0), torch.Tensor(d).unsqueeze(0)).cpu().item()
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
        im = to_img(torch.Tensor(dc).unsqueeze(0), "2D" if args['dims'] == 2 else "3D")
        imageio.imwrite(os.path.join(save_folder, "tthresh_"+args['file']+"_"+str(value)+".png"), im)

        print("Target: " +args['metric'] + " " + str(value))
        print("PSNR: " + str(rec_psnr) + " SSIM: " + str(rec_ssim))

        if(args['save_netcdf']):
            from netCDF4 import Dataset
            rootgrp = Dataset("tthresh_"+args['file']+".nc", "w", format="NETCDF4")
            rootgrp.createDimension("u")
            rootgrp.createDimension("v")
            if(args['dims'] == 3):
                rootgrp.createDimension("w")
            rootgrp.createDimension("channels", dc.shape[0])
            if(args['dims'] == 3):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v","w"))
            elif(args['dims'] == 2):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v"))
            dim_0[:] = dc[0]

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        results['rec_psnr'].append(rec_psnr)
        results['rec_ssim'].append(rec_ssim)
        results['rec_mre'].append(final_mre)
        results['rec_pwmre'].append(final_pwmre)
        results['rec_inner_mre'].append(inner_mre)
        results['rec_inner_pwmre'].append(inner_pwmre)
        if(args['save_TKE']):
            results['TKE_error'].append(0.5*((d**2).mean()-(dc**2).mean()))
        all_data['TTHRESH'] = results
        save_obj(all_data, os.path.join(save_folder, "results.pkl"))
        value += args['value_skip']
    '''
    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
    else:
        all_data = {}
    '''
    

    os.remove(args['file']+'.dat')    
    os.remove(args['file']+'.dat.tthresh.out')