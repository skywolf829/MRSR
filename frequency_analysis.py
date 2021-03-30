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
parser.add_argument('--GT',default="isomag3D_compressiontest.nc")
parser.add_argument('--NN',default="isomag3D_compressiontest_NN.nc")
parser.add_argument('--octree',default="isomag3D_compressiontest_NN_octree.nc")
parser.add_argument('--SZ',default="isomag3D_compressiontest_SZ.nc")
parser.add_argument('--device',default="cuda:0")

args = vars(parser.parse_args())

def get_circle(full_size, radius, shell_size, device):
    xx = torch.arange(0, full_size[0], dtype=torch.float, device=device).view(-1, 1,).repeat(1, full_size[1])
    yy = torch.arange(0, full_size[1], dtype=torch.float, device=device).view(1, -1).repeat(full_size[0], 1)
    circle = (((xx-(full_size[0]/2.0))**2) + ((yy-(full_size[1]/2.0))**2))**0.5
    circle = torch.logical_and(circle < (radius + int(shell_size/2)), circle > (radius - int(shell_size/2)))
    return circle

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
octree_freqs = []
NN_freqs = []
SZ_freqs = []

xs = []

GT = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['GT']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
if(len(GT.shape) == 3):
    GT_fft = fft.fftn(GT, dim=(-3, -2, -1))
elif(len(GT.shape) == 2):
    GT_fft = fft.fftn(GT, dim=(-2, -1))
del GT
if(len(GT_fft.shape) == 3):
    GT_fft = torch.roll(GT_fft, shifts=(GT_fft.shape[0]//2, GT_fft.shape[1]//2, GT_fft.shape[2]//2), dims=(-3, -2, -1))
elif(len(GT_fft.shape) == 2):
    GT_fft = torch.roll(GT_fft, shifts=(GT_fft.shape[0]//2, GT_fft.shape[1]//2), dims=(-2, -1))

n_bins = int(GT_fft.shape[0] / 2)
shell_size = 7
full_size = list(GT_fft.shape)
GT_fft = GT_fft.to("cuda:0")
device = "cuda:0"
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    if(len(GT_fft.shape) == 3):
        sphere = get_sphere(full_size, radius, shell_size, device)
    elif(len(GT_fft.shape) == 2):
        sphere = get_circle(full_size, radius, shell_size, device)
    xs.append(radius)

    GT_freqs.append(torch.abs((sphere*GT_fft).real).mean().cpu().numpy().item())
del GT_fft, sphere

device = "cuda:0"
NN = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['NN']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
if(len(NN.shape) == 3):
    NN_fft = fft.fftn(NN, dim=(-3, -2, -1))
elif(len(NN.shape) == 2):
    NN_fft = fft.fftn(NN, dim=(-2, -1))
del NN
if(len(NN_fft.shape) == 3):
    NN_fft = torch.roll(NN_fft, shifts=(NN_fft.shape[0]//2, NN_fft.shape[1]//2, NN_fft.shape[2]//2), dims=(-3, -2, -1))
elif(len(NN_fft.shape) == 2):
    NN_fft = torch.roll(NN_fft, shifts=(NN_fft.shape[0]//2, NN_fft.shape[1]//2), dims=(-2, -1))

NN_fft = NN_fft.to(device)
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    if(len(NN_fft.shape) == 3):
        sphere = get_sphere(full_size, radius, shell_size, device)
    elif(len(NN_fft.shape) == 2):
        sphere = get_circle(full_size, radius, shell_size, device)
    NN_freqs.append(torch.abs((sphere*NN_fft).real).mean().cpu().numpy().item())
del NN_fft, sphere

device = "cuda:0"
OT = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['octree']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
if(len(OT.shape) == 3):
    OT_fft = fft.fftn(OT, dim=(-3, -2, -1))
elif(len(OT.shape) == 2):
    OT_fft = fft.fftn(OT, dim=(-2, -1))
del OT
if(len(OT_fft.shape) == 3):
    OT_fft = torch.roll(OT_fft, shifts=(OT_fft.shape[0]//2, OT_fft.shape[1]//2, OT_fft.shape[2]//2), dims=(-3, -2, -1))
elif(len(OT_fft.shape) == 2):
    OT_fft = torch.roll(OT_fft, shifts=(OT_fft.shape[0]//2, OT_fft.shape[1]//2), dims=(-2, -1))

OT_fft = OT_fft.to(device)
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    if(len(OT_fft.shape) == 3):
        sphere = get_sphere(full_size, radius, shell_size, device)
    elif(len(OT_fft.shape) == 2):
        sphere = get_circle(full_size, radius, shell_size, device)
    octree_freqs.append(torch.abs((sphere*OT_fft).real).mean().cpu().numpy().item())
del OT_fft, sphere


device = "cuda:0"
SZ = torch.tensor(np.array(Dataset(os.path.join(input_folder, args['SZ']), 'r', 
format="NETCDF4")['velocity magnitude']), device=device)
if(len(SZ.shape) == 3):
    SZ_fft = fft.fftn(SZ, dim=(-3, -2, -1))
elif(len(SZ.shape) == 2):
    SZ_fft = fft.fftn(SZ, dim=(-2, -1))
del SZ
if(len(SZ_fft.shape) == 3):
    SZ_fft = torch.roll(SZ_fft, shifts=(SZ_fft.shape[0]//2, SZ_fft.shape[1]//2, SZ_fft.shape[2]//2), dims=(-3, -2, -1))
elif(len(SZ_fft.shape) == 2):
    SZ_fft = torch.roll(SZ_fft, shifts=(SZ_fft.shape[0]//2, SZ_fft.shape[1]//2), dims=(-2, -1))

SZ_fft = SZ_fft.to(device)
for i in range(0, n_bins):
    print("Bin " + str(i))
    radius = i*((full_size[0]/2) / n_bins)
    if(len(SZ_fft.shape) == 3):
        sphere = get_sphere(full_size, radius, shell_size, device)
    elif(len(SZ_fft.shape) == 2):
        sphere = get_circle(full_size, radius, shell_size, device)

    SZ_freqs.append(torch.abs((sphere*SZ_fft).real).mean().cpu().numpy().item())
del SZ_fft, sphere

fig = plt.figure()

xs = np.array(xs)
plt.plot(xs, np.array(GT_freqs), label="Raw data", color="red")
plt.plot(xs, np.array(NN_freqs), label="Ours", color="blue")
plt.plot(xs, np.array(octree_freqs), label="SR-octree", color="gray")
plt.plot(xs, np.array(SZ_freqs), label="SZ", color="green")

plt.title("Iso2D magnitude")
plt.ylabel("Power")
plt.xlabel("Wavenumber")
#plt.legend()
#plt.yscale("log")
plt.xscale("log")
print("xs")
print(xs)
print("GT_freqs")
print(GT_freqs)
print("NN freqs")
print(NN_freqs)
print("Octree freqs")
print(octree_freqs)
print("SZ frequs")
print(SZ_freqs)
fig.tight_layout()
plt.show()
plt.savefig("powerspectra.png")

plt.clf()

fig = plt.figure()
plt.plot(xs[(xs >= 10) & (xs <= 100)], np.array(GT_freqs)[(xs >= 10) & (xs <= 100)], label="Raw data", color="red")
plt.plot(xs[(xs >= 10) & (xs <= 100)], np.array(NN_freqs)[(xs >= 10) & (xs <= 100)], label="Ours", color="blue")
plt.plot(xs[(xs >= 10) & (xs <= 100)], np.array(octree_freqs)[(xs >= 10) & (xs <= 100)], label="SR-octree", color="gray")
plt.plot(xs[(xs >= 10) & (xs <= 100)], np.array(SZ_freqs)[(xs >= 10) & (xs <= 100)], label="SZ", color="green")

plt.title("Mixing3d pressure")
plt.ylabel("Power")
plt.xlabel("Wavenumber")
#plt.legend()
#plt.yscale("log")
plt.xscale("log")
fig.tight_layout()
plt.show()