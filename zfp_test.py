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

def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

def MSE(x, GT):
    return ((x-GT)**2).mean()

def PSNR(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    p = 20 * np.log10(max_diff) - 10*np.log10(MSE(x, GT))
    return p

def relative_error(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    val = np.abs(GT-x).max() / max_diff
    return val

def pw_relative_error(x, GT):
    val = np.abs(GT-x) / GT
    return val.max()


def gaussian(window_size : int, sigma : float) -> torch.Tensor:
    gauss : torch.Tensor = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x \
        in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size : torch.Tensor, channel : int) -> torch.Tensor:
    _1D_window : torch.Tensor = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window : torch.Tensor = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window : torch.Tensor = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1 : torch.Tensor, img2 : torch.Tensor, window : torch.Tensor, 
window_size : torch.Tensor, channel : int, size_average : Optional[bool] = True):
    mu1 : torch.Tensor = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 : torch.Tensor = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq : torch.Tensor = mu1.pow(2)
    mu2_sq : torch.Tensor = mu2.pow(2)
    mu1_mu2 : torch.Tensor = mu1*mu2

    sigma1_sq : torch.Tensor = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq : torch.Tensor = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 : torch.Tensor = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 : float = 0.01**2
    C2 : float= 0.03**2

    ssim_map : torch.Tensor = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ans : torch.Tensor = torch.Tensor([0])
    if size_average:
        ans = ssim_map.mean()
    else:
        ans = ssim_map.mean(1).mean(1).mean(1)
    return ans

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def to_img(input : torch.Tensor, mode : str):
    if(mode == "2D"):
        img = input[0].permute(1, 2, 0).cpu().numpy()
        img -= img.min()
        img *= (255/(img.max()+1e-6))
        img = img.astype(np.uint8)
    elif(mode == "3D"):
        img = input[0,:,:,:,int(input.shape[4]/2)].permute(1, 2, 0).cpu().numpy()
        img -= img.min()
        img *= (255/(img.max()+1e-6))
        img = img.astype(np.uint8)
    return img


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
        elif(args['dims'] == 3):      
            rec_ssim = ssim3D(torch.Tensor(dc).unsqueeze(0), torch.Tensor(d).unsqueeze(0))
        im = to_img(torch.Tensor(dc).unsqueeze(0), "2D" if args['dims'] == 2 else "3D")
        imageio.imwrite(os.path.join(save_folder, "zfp_"+args['file']+"_"+str(value)+".png"), im)

        

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        results['rec_psnr'].append(rec_psnr)
        results['rec_ssim'].append(rec_ssim)
        results['rec_mre'].append(final_mre)
        results['rec_pwmre'].append(final_pwmre)
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