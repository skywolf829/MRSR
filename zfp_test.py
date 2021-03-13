import os
import numpy as np
import imageio
import argparse
import h5py
import time
import pickle
import torch
import torch.nn.functional as F
from math import exp
from typing import Dict, List, Tuple, Optional

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
    parser.add_argument('--start_bpv',default=0.5,type=float,help='bits per value to start tests at')
    parser.add_argument('--end_bpv',default=16,type=float,help='bits per value to end tests at')
    parser.add_argument('--bpv_skip',default=0.5,type=float,help='bits per value increment by')
    parser.add_argument('--metric',default='psnr',type=str)
    

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
    results['rec_psnr'] = []
    results['rec_ssim'] = []
    results['psnrs'] = []
    results['compression_time'] = []
    results['rec_mre'] = []
    results['rec_pwmre'] = []
    results['rec_inner_mre'] = []
    results['rec_inner_pwmre'] = []
    f = h5py.File(os.path.join(input_folder, args['file']), "r")
    d = np.array(f['data'])
    f.close()
    d.tofile(args['file'] + ".dat")
    value = args['start_bpv']
    while(value < args['end_bpv']):
        data_channels = []
        f_size_kb = 0
        for i in range(args['channels']):
            d[i].tofile(args['file'] + ".dat")
            command = "zfp -f -i " + args['file'] + ".dat -z " + \
                args['file']+".dat.zfp -" + str(args['dims']) + " " + \
                str(args['nx']) + " " + str(args['ny'])
            if(args['dims'] == 3):
                command = command + " " + str(args['nz'])
            command = command + " -r " + str(value)
            start_t = time.time()
            print("Running: " + command)
            os.system(command)
            compression_time = time.time() - start_t

            f_size_kb += os.path.getsize(args['file'] + ".dat.zfp") / 1024

            command = "zfp -f -z " + args['file'] + ".dat.zfp -o " + \
                args['file'] + ".dat.zfp.out -" + str(args['dims']) + " " + \
                str(args['nx']) + " " + str(args['ny'])
            if(args['dims'] == 3):
                command = command + " " + str(args['nz'])
            command = command + " -r " + str(value) 

            os.system(command)

            dc = np.fromfile(args['file']+".dat.zfp.out")
            dc.dtype = np.float32
            if(args['dims'] == 2):
                dc = dc.reshape(args['nx'], args['ny'])
            elif(args['dims'] == 3):
                dc = dc.reshape(args['nx'], args['ny'], args['nz'])  
            data_channels.append(dc)
            command = "mv " + args['file']+".dat.zfp " + save_folder +\
                "/psnr_"+str(value)+"_"+args['file']+"_"+str(i)+".zfp"
            os.system(command)  

        dc = np.stack(data_channels)

        rec_psnr = PSNR(dc, d)
        final_mre : float = relative_error(dc, d).item()
        final_pwmre: float = pw_relative_error(dc, d).item()
        #rec_ssim = ssim(d, dc)
                
        if(args['dims'] == 2):
            rec_ssim = ssim(torch.Tensor(dc).unsqueeze(0), torch.Tensor(d).unsqueeze(0))
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20])
        elif(args['dims'] == 3):      
            rec_ssim = ssim3D(torch.Tensor(dc).unsqueeze(0), torch.Tensor(d).unsqueeze(0))
            inner_mre = relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
            inner_pwmre = pw_relative_error(dc[:,20:dc.shape[1]-20,20:dc.shape[2]-20,20:dc.shape[3]-20], 
            d[:,20:d.shape[1]-20,20:d.shape[2]-20,20:d.shape[3]-20])
        im = to_img(torch.Tensor(dc).unsqueeze(0), "2D" if args['dims'] == 2 else "3D")
        imageio.imwrite(os.path.join(save_folder, "zfp_"+args['file']+"_"+str(value)+".png"), im)

        

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        results['rec_psnr'].append(rec_psnr)
        results['rec_ssim'].append(rec_ssim)
        results['rec_mre'].append(final_mre)
        results['rec_pwmre'].append(final_pwmre)
        results['rec_inner_mre'].append(inner_mre)
        results['rec_inner_pwmre'].append(inner_pwmre)
        value += args['bpv_skip']

    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
    else:
        all_data = {}

    all_data['zfp'] = results
    save_obj(all_data, os.path.join(save_folder, "results.pkl"))

    os.remove(args['file']+'.dat')    
    #os.remove(args['file']+'.dat.zfp')
    os.remove(args['file']+'.dat.zfp.out')