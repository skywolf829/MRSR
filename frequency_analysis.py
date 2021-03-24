import torch.nn.functional as F
import torch
import os
import imageio
import argparse
from typing import Union, Tuple
from matplotlib.pyplot import cm
from math import log
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import torch.fft as fft

FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(FlowSTSR_folder_path, "TestingData")

parser = argparse.ArgumentParser(description='Test fourier space results')

parser.add_argument('--datafolder',default="")
parser.add_argument('--GT',default="mag3D_compressiontest.nc")
parser.add_argument('--NN',default="mag3D_compressiontest_NN.nc")
parser.add_argument('--SZ',default="mag3D_compressiontest_SZ.nc")
parser.add_argument('--device',default="cuda:0")

args = vars(parser.parse_args())

def get_sphere(full_size, radius, shell_size, device):
    xxx = torch.arange(0, full_size[0], dtype=torch.float, device=device).view(-1, 1,1).repeat(1, full_size[1], full_size[2])
    yyy = torch.arange(0, full_size[1], dtype=torch.float, device=device).view(1, -1,1).repeat(full_size[0], 1, full_size[2])
    zzz = torch.arange(0, full_size[2], dtype=torch.float, device=device).view(1, 1,-1).repeat(full_size[0], full_size[1], 1)
    sphere = (((xxx-(full_size[0]/2.0))**2) + ((yyy-(full_size[1]/2.0))**2) + ((zzz-(full_size[2]/2.0))**2))**0.5
    sphere = torch.logical_and(sphere < (radius + int(shell_size/2)), sphere > (radius - int(shell_size/2)))
    return sphere

def get_ks(x, y, z, xmax, ymax, zmax, device):
    xxx = torch.arange(x,device=device).type(torch.cuda.FloatTensor).view(-1, 1,1).repeat(1, y, z)
    yyy = torch.arange(y,device=device).type(torch.cuda.FloatTensor).view(1, -1,1).repeat(x, 1, z)
    zzz = torch.arange(z,device=device).type(torch.cuda.FloatTensor).view(1, 1,-1).repeat(x, y, 1)
    xxx[xxx>xmax] -= xmax*2
    yyy[yyy>ymax] -= ymax*2
    zzz[zzz>zmax] -= zmax*2
    ks = (xxx*xxx + yyy*yyy + zzz*zzz) ** 0.5
    ks = torch.round(ks).type(torch.LongTensor)
    return ks

device = args['device']
font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 18}
plt.rcParams.update(font)

GT_freqs = []
NN_freqs = []
SZ_freqs = []

xs = []

GT = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['GT']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
GT_fft = fft.fftn(GT, dim=(-3, -2, -1))
del GT
GT_fft = torch.roll(GT_fft, shifts=(GT_fft.shape[0]//2, GT_fft.shape[1]//2, GT_fft.shape[2]//2), dims=(-3, -2, -1))

n_bins = 256
shell_size = 3
full_size = list(GT_fft.shape)
GT_fft = GT_fft.to("cuda:0")
device = "cuda:0"
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    sphere = get_sphere(full_size, radius, shell_size, device)
    
    xs.append(radius)

    GT_freqs.append(torch.abs((sphere*GT_fft).real).mean().cpu().numpy())
del GT_fft

NN = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['NN']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
NN_fft = fft.fftn(NN, dim=(-3, -2, -1))
del NN
NN_fft = torch.roll(NN_fft, shifts=(NN_fft.shape[0]//2, NN_fft.shape[1]//2, NN_fft.shape[2]//2), dims=(-3, -2, -1))
NN_fft = NN_fft.to("cuda:0")
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    sphere = get_sphere(full_size, radius, shell_size, device)

    NN_freqs.append(torch.abs((sphere*NN_fft).real).mean().cpu().numpy())
del NN_fft


SZ = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['SZ']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
SZ_fft = fft.fftn(SZ, dim=(-3, -2, -1))
del SZ
SZ_fft = torch.roll(SZ_fft, shifts=(SZ_fft.shape[0]//2, SZ_fft.shape[1]//2, SZ_fft.shape[2]//2), dims=(-3, -2, -1))
SZ_fft = SZ_fft.to("cuda:0")
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    sphere = get_sphere(full_size, radius, shell_size, device)

    SZ_freqs.append(torch.abs((sphere*SZ_fft).real).mean().cpu().numpy())
del SZ_fft

fig = plt.figure()

plt.plot(xs, GT_freqs, label="Raw data")
plt.plot(xs, NN_freqs, label="Ours")
plt.plot(xs, SZ_freqs, label="SZ")

plt.title("Radially Averaged Power Spectrum Density")
plt.ylabel("Power")
plt.xlabel("Wavenumber")
plt.legend()
print("xs")
print(xs)
print("GT_freqs, NN_freqs, and SZ_freqs")
print(GT_freqs)
print()
print(NN_freqs)
print()
print(SZ_freqs)
#plt.show()
plt.savefig("powerspectra.png")