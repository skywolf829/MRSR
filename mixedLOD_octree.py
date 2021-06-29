import imageio
import numpy as np
import torch
from torch._C import ScriptModule
import torch.nn.functional as F
from math import log2
import torch.jit
from torch.nn.modules.module import T
from utility_functions import *
import time
from typing import Dict, List, Tuple, Optional
import h5py
from spatial_models import load_models
from options import load_options
import argparse
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
from utility_functions import save_obj, load_obj, ssim, ssim3D, to_img
from pytorch_memlab import LineProfiler

@torch.jit.script
class OctreeNode:
    def __init__(self, data : torch.Tensor, 
    LOD : int, depth : int, index : int):
        self.data : torch.Tensor = data 
        self.LOD : int = LOD
        self.depth : int = depth
        self.index : int = index

    def __str__(self) -> str:
        return "{ data_shape: " + str(self.data.shape) + ", " + \
        "LOD: " + str(self.LOD) + ", " + \
        "depth: " + str(self.depth) + ", " + \
        "index: " + str(self.index) + "}" 
        
    def min_width(self) -> int:
        m = self.data.shape[2]
        for i in range(3, len(self.data.shape)):
            m = min(m, self.data.shape[i])
        return m

    def size(self) -> float:
        return (self.data.element_size() * self.data.numel()) / 1024.0

@torch.jit.script
def get_location2D(full_height: int, full_width : int, depth : int, index : int) -> Tuple[int, int]:
    final_x : int = 0
    final_y : int = 0

    current_depth : int = depth
    current_index : int = index
    while(current_depth > 0):
        s_x = int(full_width / (2**current_depth))
        s_y = int(full_height / (2**current_depth))
        x_offset = s_x * int((current_index % 4) / 2)
        y_offset = s_y * (current_index % 2)
        final_x += x_offset
        final_y += y_offset
        current_depth -= 1
        current_index = int(current_index / 4)

    return (final_x, final_y)

@torch.jit.script
def get_location3D(full_width : int, full_height: int, full_depth : int, 
depth : int, index : int) -> Tuple[int, int, int]:
    final_x : int = 0
    final_y : int = 0
    final_z : int = 0

    current_depth : int = depth
    current_index : int = index
    while(current_depth > 0):
        s_x = int(full_width / (2**current_depth))
        s_y = int(full_height / (2**current_depth))
        s_z = int(full_depth / (2**current_depth))

        x_offset = s_x * int((current_index % 8) / 4)
        y_offset = s_y * int((current_index % 4) / 2)
        z_offset = s_z * (current_index % 2)
        
        final_x += x_offset
        final_y += y_offset
        final_z += z_offset
        current_depth -= 1
        current_index = int(current_index / 8)

    return (final_x, final_y, final_z)

@torch.jit.script
class OctreeNodeList:
    def __init__(self):
        self.node_list : List[OctreeNode] = []
    def append(self, n : OctreeNode):
        self.node_list.append(n)
    def insert(self, i : int, n: OctreeNode):
        self.node_list.insert(i, n)
    def pop(self, i : int) -> OctreeNode:
        return self.node_list.pop(i)
    def remove(self, item : OctreeNode) -> bool:
        found : bool = False
        i : int = 0
        while(i < len(self.node_list) and not found):
            if(self.node_list[i] is item):
                self.node_list.pop(i)
                found = True
            i += 1
        return found
    def __len__(self) -> int:
        return len(self.node_list)
    def __getitem__(self, key : int) -> OctreeNode:
        return self.node_list[key]
    def __str__(self):
        s : str = "["
        for i in range(len(self.node_list)):
            s += str(self.node_list[i])
            if(i < len(self.node_list)-1):
                s += ", "
        s += "]"
        return s
    def total_size(self):
        nbytes = 0.0
        for i in range(len(self.node_list)):
            nbytes += self.node_list[i].size()
        return nbytes 
    
    def mean(self):
        m = torch.zeros([self.node_list[0].data.shape[1]])
        dims = [0, 2, 3]
        if(len(self.node_list[0].data.shape) == 5):
            dims.append(4)
        for i in range(len(self.node_list)):
            m += self.node_list[i].data.mean(dims).cpu()
        return m / len(self.node_list)
    def max(self):
        m = self.node_list[0].data.max().cpu().item()
        for i in range(len(self.node_list)):
            m = max(m, self.node_list[i].data.max().cpu().item())
        return m

    def min(self):
        m = self.node_list[0].data.min().cpu().item()
        for i in range(len(self.node_list)):
            m = min(m, self.node_list[i].data.min().cpu().item())
        return m

def ssim_criterion(GT_image, img, min_ssim=0.6) -> float:
    if(len(GT_image.shape) == 4):
        return ssim(GT_image, img) > min_ssim
    elif(len(GT_image.shape) == 5):
        return ssim3D(GT_image, img) > min_ssim

@torch.jit.script
def MSE(x, GT) -> torch.Tensor:
    return ((x-GT)**2).mean()

@torch.jit.script
def PSNR(x, GT, max_diff : Optional[torch.Tensor] = None) -> torch.Tensor:
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    p = 20 * torch.log10(torch.tensor(max_diff)) - 10*torch.log10(MSE(x, GT))
    return p

@torch.jit.script
def relative_error(x, GT, max_diff : Optional[torch.Tensor] = None) -> torch.Tensor:
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    val = torch.abs(GT-x).max() / max_diff
    return val

@torch.jit.script
def pw_relative_error(x, GT) -> torch.Tensor:
    val = torch.abs(torch.abs(GT-x) / GT)
    return val.max()

@torch.jit.script
def psnr_criterion(GT_image, img, min_PSNR : torch.Tensor,
max_diff : Optional[torch.Tensor] = None) -> torch.Tensor:
    meets = PSNR(img, GT_image, max_diff) > min_PSNR
    return meets

@torch.jit.script
def mse_criterion(GT_image, img, max_mse : torch.Tensor) -> torch.Tensor:
    return MSE(img, GT_image) < max_mse

@torch.jit.script
def maximum_relative_error(GT_image, img, max_e : torch.Tensor, 
max_diff : Optional[torch.Tensor] = None) -> torch.Tensor:
    return relative_error(img, GT_image, max_diff) < max_e

@torch.jit.script
def maximum_pw_relative_error(GT_image, img, max_e : torch.Tensor) -> torch.Tensor:
    return pw_relative_error(img, GT_image) < max_e

@torch.jit.script
def bilinear_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    align_corners=False, mode='bilinear')
    return img

@torch.jit.script
def trilinear_upscale(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = F.interpolate(vol, scale_factor=float(scale_factor), 
    align_corners=False, mode='trilinear')
    return vol

@torch.jit.script
def bicubic_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    align_corners=False, mode='bicubic')
    img = img.clamp_(0.0, 255.0)
    return img

def model_upscale(input : torch.Tensor, scale_factor : int, 
    models, lod : int) -> torch.Tensor:
    with torch.no_grad():
        final_out = input
        while(scale_factor > 1):
            final_out = models[len(models)-lod](final_out)
            scale_factor = int(scale_factor / 2)
            lod -= 1
    return final_out

@torch.jit.script
def point_upscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    upscaled_img = torch.zeros([img.shape[0], img.shape[1],
    int(img.shape[2]*scale_factor), 
    int(img.shape[3]*scale_factor)]).to(img.device)
    
    for x in range(img.shape[2]):
        for y in range(img.shape[3]):
            upscaled_img[:,:,x*scale_factor:(x+1)*scale_factor, 
            y*scale_factor:(y+1)*scale_factor] = img[:,:,x,y].view(img.shape[0], img.shape[1], 1, 1).repeat(1, 1, scale_factor, scale_factor)
    return upscaled_img

@torch.jit.script
def point_upscale3D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    upscaled_img = torch.zeros([img.shape[0], img.shape[1],
    int(img.shape[2]*scale_factor), 
    int(img.shape[3]*scale_factor),
    int(img.shape[4]*scale_factor)]).to(img.device)
    
    for x in range(img.shape[2]):
        for y in range(img.shape[3]):
            for z in range(img.shape[4]):
                upscaled_img[:,:,
                x*scale_factor:(x+1)*scale_factor, 
                y*scale_factor:(y+1)*scale_factor,
                z*scale_factor:(z+1)*scale_factor] = \
                    img[:,:,x,y,z].view(img.shape[0], img.shape[1], 1, 1, 1).repeat(1, 1, scale_factor, scale_factor, scale_factor)
    return upscaled_img

@torch.jit.script
def nearest_neighbor_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    mode='nearest')
    return img

@torch.jit.script
def bilinear_downscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=(1/scale_factor), align_corners=True, mode='bilinear')
    return img

@torch.jit.script
def avgpool_downscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = AvgPool2D(img, scale_factor)
    return img

@torch.jit.script
def avgpool_downscale3D(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = AvgPool3D(vol, scale_factor)
    return vol

@torch.jit.script
def subsample_downscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = img[:,:, ::scale_factor, ::scale_factor]
    return img

@torch.jit.script
def subsample_downscale3D(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = vol[:,:, ::scale_factor, ::scale_factor, ::scale_factor]
    return vol

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
            input_volume.shape[3]*2, input_volume.shape[4]*2],
            dtype=torch.float32).to(devices[0])
        
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

def generate_patch(z,z_stop,y,y_stop,x,x_stop,available_gpus):

    device = None
    while(device is None):        
        device, generator, input_volume = available_gpus.get_next_available()
        time.sleep(1)
    #print("Starting SR on device " + device)
    with torch.no_grad():
        result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])
    return result,z,z_stop,y,y_stop,x,x_stop,device

class UpscalingMethod(nn.Module):
    def __init__(self, method : str, device : str, model_name : Optional[str] = None,
        distributed : Optional[bool] = False):
        super(UpscalingMethod, self).__init__()
        self.method : str = method
        self.device : str = device
        self.models = []
        self.distributed = distributed
        self.devices = []
        if(self.method == "model"):            
            self.load_models(model_name, device, distributed)

    def change_method(self, method : str, model_name : Optional[str] = None, 
    device : Optional[str] = None, distributed : Optional[bool] = False):
        self.method = method
        if(method == "model" and len(self.models) == 0):
            self.load_models(model_name, device, distributed)
        
    def load_models(self, model_name, device, distributed):
        with torch.no_grad():
            options = load_options("SavedModels/"+model_name)
            torch_models, discs = load_models(options, "cpu")
            for i in range(len(torch_models)):
                torch_models[i] = torch_models[i].to(device)
                del(discs[0])
                #self.models.append(torch_models[i])
                
                print("Tracing model " + str(i) + " with input size " + str(torch_models[i].get_input_shape()))
                self.models.append(torch.jit.trace(torch_models[i], 
                torch.zeros(torch_models[0].get_input_shape()).to(device)))
                
        torch.cuda.empty_cache()
        if(distributed and torch.cuda.device_count() > 1):
            for i in range(torch.cuda.device_count()):
                self.devices.append("cuda:"+str(i))
        print("Loaded models")

    def forward(self, in_frame : torch.Tensor, scale_factor : float,
    lod : Optional[int] = None) -> torch.Tensor:
        with torch.no_grad():
            up = torch.empty([1],device=in_frame.device)
            if(self.method == "bilinear"):
                up = bilinear_upscale(in_frame, scale_factor)
            elif(self.method == "bicubic"):
                up = bicubic_upscale(in_frame, scale_factor)
            elif(self.method == "point2D"):
                up = point_upscale2D(in_frame, scale_factor)
            elif(self.method == "point3D"):
                up = point_upscale3D(in_frame, scale_factor)
            elif(self.method == "nearest"):
                up = nearest_neighbor_upscale(in_frame, scale_factor)
            elif(self.method == "trilinear"):
                up = trilinear_upscale(in_frame, scale_factor)
            elif(self.method == "model"):
                up = in_frame
                while(scale_factor > 1):
                    if not self.distributed:
                        up = self.models[len(self.models)-lod](up)
                    else:
                        up = generate_by_patch_parallel(self.models[len(self.models)-lod], 
                            up, 140, 10, self.devices)
                    scale_factor = int(scale_factor / 2)
                    lod -= 1
            else:
                print("No support for upscaling method: " + str(self.method))
        return up

@torch.jit.script
def downscale(method: str, img: torch.Tensor, scale_factor: int) -> torch.Tensor:
    with torch.no_grad():
        down = torch.zeros([1])        
        if(method == "bilinear"):
            down = bilinear_downscale(img, scale_factor)
        elif(method == "subsample2D"):
            down = subsample_downscale2D(img, scale_factor)
        elif(method == "subsample3D"):
            down = subsample_downscale3D(img, scale_factor)
        elif(method == "avgpool2D"):
            down = avgpool_downscale2D(img, scale_factor)
        elif(method == "avgpool3D"):
            down = avgpool_downscale3D(img, scale_factor)
        else:
            print("No support for downscaling method: " + str(method))
    return down

def criterion_met(method: str, value: torch.Tensor, 
a: torch.Tensor, b: torch.Tensor, max_diff : Optional[torch.Tensor] = None) -> torch.Tensor:
    passed : torch.Tensor = torch.empty([1],device=a.device)
    with torch.no_grad():
        if(method == "psnr"):
            passed = psnr_criterion(a, b, value, max_diff)
        elif(method == "mse"):
            passed = mse_criterion(a, b, value)
        elif(method == "mre"):
            passed = maximum_relative_error(a, b, value, max_diff)
        elif(method == "pw_mre"):
            passed = maximum_pw_relative_error(a, b, value)
        elif(method == "ssim"):
            passed = ssim_criterion(a, b, value)
        else:
            print("No support for criterion: " + str(method))
    return passed

@torch.jit.script
def nodes_to_downscaled_levels(nodes : OctreeNodeList, full_shape : List[int],
    max_LOD : int, downscaling_technique: str, device : str, 
    data_levels: List[torch.Tensor], mask_levels:List[torch.Tensor],
    data_downscaled_levels: List[torch.Tensor], mask_downscaled_levels:List[torch.Tensor],
    mode : str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    mask_downscaled_levels[0][:] = mask_levels[0][:]
    data_downscaled_levels[0][:] = data_levels[0][:]

    curr_LOD = 1
    #imageio.imwrite("data_"+str(i+1)+"_filled_in.png", data_downscaled_levels[-1].cpu().numpy().astype(np.uint8)) 
    #imageio.imwrite("mask_"+str(i+1)+"_filled_in.png", mask_downscaled_levels[-1].cpu().numpy().astype(np.uint8)*255)
    while curr_LOD <= max_LOD:

        data_down = downscale(downscaling_technique, 
        data_downscaled_levels[curr_LOD-1], 2)
        mask_down = mask_downscaled_levels[curr_LOD-1][:,:,::2,::2]
        if(mode == "3D"):
            mask_down = mask_downscaled_levels[curr_LOD-1][:,:,::2,::2,::2]

        #imageio.imwrite("data_"+str(i)+"_downscaled.png", data_down.cpu().numpy().astype(np.uint8))        
        #imageio.imwrite("mask_"+str(i)+"_downscaled.png", mask_down.cpu().numpy().astype(np.uint8)*255)

        data_downscaled_levels[curr_LOD] = data_down + data_levels[curr_LOD]
        mask_downscaled_levels[curr_LOD] = mask_down + mask_levels[curr_LOD]

        #imageio.imwrite("data_"+str(i)+"_filldedin.png", data_downscaled_levels[i].cpu().numpy().astype(np.uint8))        
        #imageio.imwrite("mask_"+str(i)+"_filledin.png", mask_downscaled_levels[i].cpu().numpy().astype(np.uint8)*255)

        curr_LOD += 1
    return data_downscaled_levels, mask_downscaled_levels

def nodes_to_full_img(nodes: OctreeNodeList, full_shape: List[int], 
    max_LOD : int, upscale : UpscalingMethod, 
    downscaling_technique : str, device : str, 
    data_levels: List[torch.Tensor], mask_levels:List[torch.Tensor],
    data_downscaled_levels: List[torch.Tensor], mask_downscaled_levels:List[torch.Tensor],
    mode : str) -> torch.Tensor:

    data_downscaled_levels, mask_downscaled_levels = nodes_to_downscaled_levels(nodes, 
    full_shape, max_LOD, downscaling_technique,
    device, data_levels, mask_levels, data_downscaled_levels, 
    mask_downscaled_levels, mode)
    
    curr_LOD = max_LOD

    full_img = data_downscaled_levels[curr_LOD]
    while(curr_LOD > 0):
        
        full_img = upscale(full_img, 2, curr_LOD)
        torch.cuda.synchronize()
        curr_LOD -= 1

        full_img *=  (~mask_downscaled_levels[curr_LOD])
        data_downscaled_levels[curr_LOD] *= mask_downscaled_levels[curr_LOD] 
        full_img += data_downscaled_levels[curr_LOD]
    while(len(data_downscaled_levels) > 0):
        del data_downscaled_levels[0]
        del mask_downscaled_levels[0]
    return full_img

def nodes_to_full_img_debug(nodes: OctreeNodeList, full_shape: List[int], 
max_LOD : int, upscale : UpscalingMethod, 
downscaling_technique : str, device : str, mode : str) -> Tuple[torch.Tensor, torch.Tensor]:
    full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3]]).to(device)
    if(mode == "3D"):
        full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3], full_shape[4]]).to(device)
    # palette here: https://www.pinterest.com/pin/432978951683169251/?d=t&mt=login
    # morandi colors
    # https://colorbrewer2.org/#type=sequential&scheme=YlOrRd&n=6
    cmap : List[torch.Tensor] = [
        torch.tensor([[255,255,178]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[254,217,118]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[254,178,76]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[253,141,60]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[252,78,42]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[227,26,28]], dtype=nodes[0].data.dtype, device=device),        
        torch.tensor([[177,0,38]], dtype=nodes[0].data.dtype, device=device)
    ]
    for i in range(len(cmap)):
        cmap[i] = cmap[i].unsqueeze(2).unsqueeze(3)
        if(mode == "3D"):
            cmap[i] = cmap[i].unsqueeze(4)

    for i in range(len(nodes)):
        curr_node = nodes[i]
        if(mode == "2D"):
            x_start, y_start = get_location2D(full_shape[2], full_shape[3], curr_node.depth, curr_node.index)
            s : int = curr_node.LOD
            full_img[:,:,
                int(x_start): \
                int(x_start)+ \
                    int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                int(y_start): \
                int(y_start)+ \
                    int((curr_node.data.shape[3]*(2**curr_node.LOD)))
            ] = torch.zeros([full_shape[0], 3, 
            curr_node.data.shape[2]*(2**curr_node.LOD),
            curr_node.data.shape[3]*(2**curr_node.LOD)])
            full_img[:,:,
                int(x_start)+1: \
                int(x_start)+ \
                    int((curr_node.data.shape[2]*(2**curr_node.LOD)))-1,
                int(y_start)+1: \
                int(y_start)+ \
                    int((curr_node.data.shape[3]*(2**curr_node.LOD)))-1
            ] = cmap[s].repeat(full_shape[0], 1, 
            int((curr_node.data.shape[2]*(2**curr_node.LOD)))-2, 
            int((curr_node.data.shape[3]*(2**curr_node.LOD)))-2)
        elif(mode == "3D"):
            x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4],
            curr_node.depth, curr_node.index)
            s : int = curr_node.LOD
            full_img[:,:,
                int(x_start): \
                int(x_start)+ \
                    int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                int(y_start): \
                int(y_start)+ \
                    int((curr_node.data.shape[3]*(2**curr_node.LOD))),
                int(z_start): \
                int(z_start)+ \
                    int((curr_node.data.shape[4]*(2**curr_node.LOD)))
            ] = torch.zeros([full_shape[0], 3, 
            curr_node.data.shape[2]*(2**curr_node.LOD),
            curr_node.data.shape[3]*(2**curr_node.LOD),
            curr_node.data.shape[4]*(2**curr_node.LOD)])
            full_img[:,:,
                int(x_start)+1: \
                int(x_start)+ \
                    int((curr_node.data.shape[2]*(2**curr_node.LOD)))-1,
                int(y_start)+1: \
                int(y_start)+ \
                    int((curr_node.data.shape[3]*(2**curr_node.LOD)))-1,
                int(z_start)+1: \
                int(z_start)+ \
                    int((curr_node.data.shape[4]*(2**curr_node.LOD)))-1
            ] = cmap[s].repeat(full_shape[0], 1, 
            int((curr_node.data.shape[2]*(2**curr_node.LOD)))-2, 
            int((curr_node.data.shape[3]*(2**curr_node.LOD)))-2,
            int((curr_node.data.shape[4]*(2**curr_node.LOD)))-2)
    cmap_img_height : int = 64
    cmap_img_width : int = 512
    cmap_img = torch.zeros([cmap_img_width, cmap_img_height, 3], dtype=torch.float, device=device)
    y_len : int = int(cmap_img_width / len(cmap))
    for i in range(len(cmap)):
        y_start : int = i * y_len
        y_end : int = (i+1) * y_len
        cmap_img[y_start:y_end, :, :] = torch.squeeze(cmap[i])

    return full_img, cmap_img

def nodes_to_full_img_seams(nodes: OctreeNodeList, full_shape: List[int], 
upscale : UpscalingMethod, device: str, mode : str):
    full_img = torch.zeros(full_shape).to(device)
    
    # 1. Fill in known data
    for i in range(len(nodes)):
        curr_node = nodes[i]
        if(mode == "2D"):
            x_start, y_start = get_location2D(full_shape[2], full_shape[3], curr_node.depth, curr_node.index)
            img_part = upscale(curr_node.data, 2**curr_node.LOD, curr_node.LOD)
            full_img[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3]] = img_part
        elif(mode == "3D"):
            x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], curr_node.depth, curr_node.index)
            img_part = upscale(curr_node.data, 2**curr_node.LOD, curr_node.LOD)
            full_img[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3],z_start:z_start+img_part.shape[4]] = img_part
    
    return full_img

@torch.jit.script
def remove_node_from_data_caches(node: OctreeNode, full_shape: List[int],
data_levels: List[torch.Tensor], mask_levels: List[torch.Tensor], mode : str):
    curr_ds_ratio = (2**node.LOD)
    if(mode == "2D"):
        x_start, y_start = get_location2D(full_shape[2], full_shape[3], node.depth, node.index)
        ind = node.LOD
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3]
        ] = 0
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3]
        ] = 0
    elif(mode == "3D"):
        x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], node.depth, node.index)
        ind = node.LOD
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = 0
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = 0

@torch.jit.script
def add_node_to_data_caches(node: OctreeNode, full_shape: List[int],
data_levels: List[torch.Tensor], mask_levels: List[torch.Tensor], mode : str):
    curr_ds_ratio = (2**node.LOD)
    if(mode == "2D"):
        x_start, y_start = get_location2D(full_shape[2], full_shape[3], node.depth, node.index)
        ind = node.LOD 
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3]
        ] = node.data
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
        ] = 1
    elif(mode == "3D"):
        x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], node.depth, node.index)
        ind = node.LOD
        
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = node.data
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = 1

@torch.jit.script
def create_caches_from_nodelist(nodes: OctreeNodeList, 
full_shape : List[int], max_LOD: int, device: str, mode : str) -> \
Tuple[List[torch.Tensor], List[torch.Tensor], 
List[torch.Tensor], List[torch.Tensor]]:
    data_levels: List[torch.Tensor] = []
    mask_levels: List[torch.Tensor] = []
    data_downscaled_levels: List[torch.Tensor] = []
    mask_downscaled_levels: List[torch.Tensor] = []
    curr_LOD = 0
    
    curr_shape : List[int] = [full_shape[0], full_shape[1], full_shape[2], full_shape[3]]
    if(mode == "3D"):
        curr_shape = [full_shape[0], full_shape[1], full_shape[2], full_shape[3], full_shape[4]]
    while(curr_LOD <= max_LOD):
        data_levels.append(torch.zeros(curr_shape, dtype=torch.float32).to(device))
        data_downscaled_levels.append(torch.zeros(curr_shape, dtype=torch.float32).to(device))
        mask_levels.append(torch.zeros(curr_shape, dtype=torch.bool).to(device))
        mask_downscaled_levels.append(torch.zeros(curr_shape, dtype=torch.bool).to(device))
        curr_shape[2] = int(curr_shape[2] / 2)
        curr_shape[3] = int(curr_shape[3] / 2)  
        if(mode == "3D"):
            curr_shape[4] = int(curr_shape[4] / 2)
        curr_LOD += 1
        
    for i in range(len(nodes)):
        add_node_to_data_caches(nodes[i], full_shape,
        data_levels, mask_levels, mode)
    
    return data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels

def mixedLOD_octree_SR_compress(
    nodes : OctreeNodeList, GT_image : torch.Tensor, 
    criterion: str, criterion_value : float,
    upscale : UpscalingMethod, downscaling_technique: str,
    min_chunk_size : int, max_LOD : int, 
    device : str, mode : str
    ) -> OctreeNodeList:
    node_indices_to_check = [ 0 ]
    nodes_checked = 0
    full_shape = nodes[0].data.shape
    max_diff = GT_image.max() - GT_image.min()
    allowed_error = torch.Tensor([criterion_value]).to(device)

    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
        create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)
    
    while(len(node_indices_to_check) > 0): 
        #print(nodes_checked)
        nodes_checked += 1
        i = node_indices_to_check.pop(0)
        n = nodes[i]

        t = time.time()

        # Check if we can downsample this node
        remove_node_from_data_caches(n, full_shape, data_levels, mask_levels, mode)
        n.LOD = n.LOD + 1
        original_data = n.data.clone()
        downsampled_data = downscale(downscaling_technique,n.data,2)
        n.data = downsampled_data
        add_node_to_data_caches(n, full_shape, data_levels, mask_levels, mode)
        
        #print("Downscaling time : " + str(time.time() - t))
        t = time.time()

        new_img = nodes_to_full_img(nodes, full_shape, max_LOD, 
        upscale, downscaling_technique,
        device, data_levels, mask_levels, data_downscaled_levels, 
        mask_downscaled_levels, mode)
        torch.cuda.synchronize()
        #print("Upscaling time : " + str(time.time() - t))
        
        # If criterion not met, reset data and stride, and see
        # if the node is large enough to split into subnodes
        # Otherwise, we keep the downsample, and add the node back as a 
        # leaf node
        t = time.time()


        met = criterion_met(criterion, allowed_error, GT_image, new_img, max_diff).item()

        if(not met):
            #print("If statement : " + str(time.time() - t))
            t = time.time()
            remove_node_from_data_caches(n, full_shape, data_levels, mask_levels, mode)
            n.data = original_data
            n.LOD = n.LOD - 1

            #print("Node removed time : " + str(time.time() - t))
            t = time.time()
            if(n.min_width()*(2**n.LOD) > min_chunk_size*2 and
                n.min_width() > 2):
                k = 0
                while k < len(node_indices_to_check):
                    if(node_indices_to_check[k] > i):
                        node_indices_to_check[k] -= 1                
                    #if(node_indices_to_check[k] == i):
                    #    node_indices_to_check.pop(k)
                    #    k -= 1
                    k += 1

                nodes.pop(i)
                k = 0

                for x_quad_start in range(0, n.data.shape[2], int(n.data.shape[2]/2)):
                    for y_quad_start in range(0, n.data.shape[3], int(n.data.shape[3]/2)):
                        if(mode == "3D"):
                            for z_quad_start in range(0, n.data.shape[4], int(n.data.shape[4]/2)):
                                n_quad = OctreeNode(
                                    n.data[:,:,
                                        x_quad_start:x_quad_start+int(n.data.shape[2]/2),
                                        y_quad_start:y_quad_start+int(n.data.shape[3]/2),
                                        z_quad_start:z_quad_start+int(n.data.shape[4]/2)].clone(),
                                    n.LOD,
                                    n.depth+1,
                                    n.index*8 + k
                                )
                                add_node_to_data_caches(n_quad, full_shape, data_levels, mask_levels, mode)
                                nodes.append(n_quad)
                                node_indices_to_check.append(len(nodes)-1) 
                                k += 1     
                        elif(mode == "2D"):
                            n_quad = OctreeNode(
                                n.data[:,:,
                                    x_quad_start:x_quad_start+int(n.data.shape[2]/2),
                                    y_quad_start:y_quad_start+int(n.data.shape[3]/2)].clone(),
                                n.LOD,
                                n.depth+1,
                                n.index*4 + k
                            )
                            add_node_to_data_caches(n_quad, full_shape, data_levels, mask_levels, mode)
                            nodes.append(n_quad)
                            node_indices_to_check.append(len(nodes)-1) 
                            k += 1       
            
            else:
                add_node_to_data_caches(n, full_shape, data_levels, mask_levels, mode) 
            #print("Node split time : " + str(time.time() - t))
            t = time.time()      
        else:
            if(n.LOD < max_LOD and 
                n.min_width()*(2**n.LOD) > min_chunk_size and
                n.min_width() > 1):
                node_indices_to_check.append(i)
        
    while(0 < len(data_levels)):
        del data_levels[0]
        del mask_levels[0]
        del data_downscaled_levels[0]
        del mask_downscaled_levels[0]
    #print("Nodes traversed: " + str(nodes_checked))
    return nodes            

def nondynamic_SR_compress(
    nodes : OctreeNodeList, GT_image : torch.Tensor, 
    criterion: str, criterion_value : float,
    upscale : UpscalingMethod, downscaling_technique: str,
    min_chunk_size : int, max_LOD : int, 
    device : str, mode : str
    ) -> OctreeNodeList:
    node_indices_to_check = [ 0 ]
    nodes_checked = 0
    full_shape = nodes[0].data.shape
    max_diff = GT_image.max() - GT_image.min()
    allowed_error = torch.Tensor([criterion_value]).to(device)

    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
        create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)
    
    while(len(node_indices_to_check) > 0): 
        #print(nodes_checked)
        nodes_checked += 1
        i = node_indices_to_check.pop(0)
        n = nodes[i]

        # Check if we can downsample this node
        remove_node_from_data_caches(n, full_shape, data_levels, mask_levels, mode)
        n.LOD = n.LOD + 1
        original_data = n.data.clone()
        downsampled_data = downscale(downscaling_technique,n.data,2)
        n.data = downsampled_data
        add_node_to_data_caches(n, full_shape, data_levels, mask_levels, mode)
        

        new_img = nodes_to_full_img(nodes, full_shape, max_LOD, 
        upscale, downscaling_technique,
        device, data_levels, mask_levels, data_downscaled_levels, 
        mask_downscaled_levels, mode)
        torch.cuda.synchronize()
        #print("Upscaling time : " + str(time.time() - t))
        
        # If criterion not met, reset data and stride, and see
        # if the node is large enough to split into subnodes
        # Otherwise, we keep the downsample, and add the node back as a 
        # leaf node

        met = criterion_met(criterion, allowed_error, GT_image, new_img, max_diff)

        if(not met):
            #print("If statement : " + str(time.time() - t))
            t = time.time()
            remove_node_from_data_caches(n, full_shape, data_levels, mask_levels, mode)
            n.data = original_data
            n.LOD = n.LOD - 1  
        else:
            if(n.LOD < max_LOD and 
                n.min_width()*(2**n.LOD) > min_chunk_size and
                n.min_width() > 1):
                node_indices_to_check.append(i)
    #print("Nodes traversed: " + str(nodes_checked))
    return nodes            

def compress_nodelist(nodes: OctreeNodeList, full_size : List[int], 
min_chunk_size: int, device : str, mode : str) -> OctreeNodeList:

    min_width : int = full_size[2]
    for i in range(3, len(full_size)):
        min_width = min(min_width, full_size[i])

    current_depth = nodes[0].depth

    # dict[depth -> LOD -> group parent index -> list]
    groups : Dict[int, Dict[int, Dict[int, Dict[int, OctreeNode]]]] = {}

    magic_num : int = 4 if mode == "2D" else 8
    for i in range(len(nodes)):
        d : int = nodes[i].depth
        l : int = nodes[i].LOD
        group_parent_index : int = int(nodes[i].index / magic_num)
        n_index : int = nodes[i].index % magic_num
        current_depth = max(current_depth, d)
        if(d not in groups.keys()):
            groups[d] = {}
        if(l not in groups[d].keys()):
            groups[d][l] = {}
        if(group_parent_index not in groups[d][l].keys()):
            groups[d][l][group_parent_index] = {}
        groups[d][l][group_parent_index][n_index] = nodes[i]

            
    while(current_depth  > 0):
        if(current_depth in groups.keys()):
            for lod in groups[current_depth].keys():
                for parent in groups[current_depth][lod].keys():
                    group = groups[current_depth][lod][parent]
                    if(len(group) == magic_num):
                        if(mode == "2D"):
                            new_data = torch.zeros([
                                group[0].data.shape[0],
                                group[0].data.shape[1],
                                group[0].data.shape[2]*2, 
                                group[0].data.shape[3]*2], device=device, 
                            dtype=group[0].data.dtype)
                            new_data[:,:,:group[0].data.shape[2],
                                    :group[0].data.shape[3]] = \
                                group[0].data

                            new_data[:,:,:group[0].data.shape[2],
                                    group[0].data.shape[3]:] = \
                                group[1].data

                            new_data[:,:,group[0].data.shape[2]:,
                                    :group[0].data.shape[3]] = \
                                group[2].data

                            new_data[:,:,group[0].data.shape[2]:,
                                    group[0].data.shape[3]:] = \
                                group[3].data
                            
                            new_node = OctreeNode(new_data, group[0].LOD, 
                            group[0].depth-1, parent)
                            nodes.append(new_node)
                            nodes.remove(group[0])
                            nodes.remove(group[1])
                            nodes.remove(group[2])
                            nodes.remove(group[3])
                            d = current_depth-1
                            l = lod
                            if(d not in groups.keys()):
                                groups[d] = {}
                            if(lod not in groups[d].keys()):
                                groups[d][l] = {}
                            if(int(parent/4) not in groups[d][l].keys()):
                                groups[d][l][int(parent/4)] = {}
                            groups[d][l][int(parent/4)][new_node.index % 4] = new_node
                        elif(mode == "3D"):
                            new_data = torch.zeros([
                                group[0].data.shape[0],
                                group[0].data.shape[1],
                                group[0].data.shape[2]*2, 
                                group[0].data.shape[3]*2,
                                group[0].data.shape[4]*2], device=device, 
                            dtype=group[0].data.dtype)
                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    :group[0].data.shape[3],
                                    :group[0].data.shape[4]] = \
                                group[0].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    :group[0].data.shape[3],
                                    group[0].data.shape[4]:] = \
                                group[1].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    group[0].data.shape[3]:,
                                    :group[0].data.shape[4]] = \
                                group[2].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    group[0].data.shape[3]:,
                                    group[0].data.shape[4]:] = \
                                group[3].data

                            new_data[:,:,
                                    group[4].data.shape[2]:,
                                    :group[0].data.shape[3],
                                    :group[0].data.shape[4]] = \
                                group[4].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    :group[0].data.shape[3],
                                    group[0].data.shape[4]:] = \
                                group[5].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    group[0].data.shape[3]:,
                                    :group[0].data.shape[4]] = \
                                group[6].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    group[0].data.shape[3]:,
                                    group[0].data.shape[4]:] = \
                                group[7].data
                            
                            new_node = OctreeNode(new_data, group[0].LOD, 
                            group[0].depth-1, int(group[0].index / 8))
                            nodes.append(new_node)
                            nodes.remove(group[0])
                            nodes.remove(group[1])
                            nodes.remove(group[2])
                            nodes.remove(group[3])
                            nodes.remove(group[4])
                            nodes.remove(group[5])
                            nodes.remove(group[6])
                            nodes.remove(group[7])
                            d = current_depth-1
                            l = lod
                            if(d not in groups.keys()):
                                groups[d] = {}
                            if(lod not in groups[d].keys()):
                                groups[d][l] = {}
                            if(int(parent/8) not in groups[d][l].keys()):
                                groups[d][l][int(parent/8)] = {}
                            groups[d][l][int(parent/8)][new_node.index % 8] = new_node
                        
        current_depth -= 1
    return nodes

def nodelist_to_h5(nodes : OctreeNodeList, name : str):
    f = h5py.File(name, "w")
    for i in range(len(nodes)):
        d = f.create_dataset(str(i), data=nodes[i].data.cpu().numpy(),
        compression="gzip", compression_opts=9)
        d.attrs['index'] = nodes[i].index
        d.attrs['depth'] = nodes[i].depth
        d.attrs['LOD'] = nodes[i].LOD
    f.close()

def h5_to_nodelist(name: str, device : str):
    f = h5py.File(name, 'r')
    nodes : OctreeNodeList = OctreeNodeList()
    for k in f.keys():
        n : OctreeNode = OctreeNode(
            torch.Tensor(f[k], device=device),
            f[k].attrs['LOD'],
            f[k].attrs['depth'],
            f[k].attrs['index']
        )
    return nodes

'''
def sz_compress_nodelist(nodes: OctreeNodeList, full_shape, max_LOD,
downscaling_technique, device, mode,
folder : str, name : str, metric : str, value : float):
    
    nearest_upscale = UpscalingMethod("nearest", device)
    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

    full_im = nodes_to_full_img(nodes, full_shape, max_LOD, nearest_upscale,
    downscaling_technique, device, data_levels, mask_levels,
    data_downscaled_levels, mask_downscaled_levels, mode)
    min_LOD = max_LOD
    for i in range(len(nodes)):
        min_LOD = min(nodes[i].LOD, min_LOD)
    if(min_LOD > 0):
        if(mode == "2D"):
            full_im = subsample_downscale2D(full_im, int(2**min_LOD))
        elif(mode == "3D"):
            full_im = subsample_downscale3D(full_im, int(2**min_LOD))
    temp_folder_path = os.path.join(folder, "Temp")
    save_location = os.path.join(folder, name +".tar.gz")
    if(not os.path.exists(temp_folder_path)):
        os.makedirs(temp_folder_path)
    
    for i in range(full_im.shape[1]):
        d = full_im.cpu().numpy()[0,i]
        d_loc = os.path.join(temp_folder_path,"nn_data_"+str(i)+".dat")
        ndims = len(d.shape)
        print(d.shape)
        d.tofile(d_loc)
        command = "sz -z -f -i " + d_loc + " -" + str(ndims) + " " + \
            str(d.shape[0]) + " " + str(d.shape[1])
        if(ndims == 3):
            command = command + " " + str(d.shape[2])
        
        if(min_LOD == 0):
            if(metric == "psnr"):
                command = command + " -M PSNR -S " + str(value)
            elif(metric == "mre"):
                command = command + " -M REL -R " + str(value)
            elif(metric == "pw_mre"):
                command = command + " -M PW_REL -P " + str(value)
        else:
            command = command + " -M PW_REL -P 0.001"
        
        #command = command + " -M PW_REL -P 0.001"
        max_diff = nodes.max() - nodes.min()
        REL_target = (max_diff / (10**(value/10))) **0.5
        #command = command + " -M REL -R " + str(REL_target)
        #command = command + " -M REL -R 0.0025"
        command = command + " -M PW_REL -P " + str(REL_target)
        print(command)
        os.system(command)
        os.system("rm " + d_loc)
    
    del full_im

    metadata : List[int] = []
    metadata.append(min_LOD)
    metadata.append(len(full_shape))
    metadata.append(full_shape[0])
    metadata.append(full_shape[1])
    for i in range(2,len(full_shape)):
        metadata.append(int(full_shape[i]/(2**min_LOD)))
    for i in range(len(nodes)):
        metadata.append(nodes[i].depth)
        metadata.append(nodes[i].index)
        metadata.append(nodes[i].LOD)

    metadata = np.array(metadata, dtype=int)
    metadata.tofile(os.path.join(temp_folder_path, "metadata"))

    os.system("tar -cjvf " + save_location + " -C " + folder + " Temp")
    os.system("rm -r " + temp_folder_path)
'''
def zfp_compress_nodelist(nodes: OctreeNodeList, full_shape, max_LOD,
downscaling_technique, device, mode,
folder : str, name : str):
    
    nearest_upscale = UpscalingMethod("nearest", device)
    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

    full_im = nodes_to_full_img(nodes, full_shape, max_LOD, nearest_upscale,
    downscaling_technique, device, data_levels, mask_levels,
    data_downscaled_levels, mask_downscaled_levels, mode)
    min_LOD = max_LOD
    for i in range(len(nodes)):
        min_LOD = min(nodes[i].LOD, min_LOD)
    if(min_LOD > 0):
        if(mode == "2D"):
            full_im = subsample_downscale2D(full_im, int(2**min_LOD))
        elif(mode == "3D"):
            full_im = subsample_downscale3D(full_im, int(2**min_LOD))
    temp_folder_path = os.path.join(folder, "Temp")
    save_location = os.path.join(folder, name +".tar.gz")
    if(not os.path.exists(temp_folder_path)):
        os.makedirs(temp_folder_path)
    

    d = full_im.cpu().numpy()[0,0]
    d_loc = os.path.join(temp_folder_path,"nn_data.dat")
    ndims = len(d.shape)
    print(d.shape)
    d.tofile(d_loc)
    command = "zfp -f -R -i " + d_loc + " -o " +\
        d_loc + ".zfp" + " -" + str(ndims) + " " + \
        str(d.shape[0]) + " " + str(d.shape[1])
    if(ndims == 3):
        command = command + " " + str(d.shape[2])
    print(command)
    os.system(command)
    os.system("rm " + d_loc)
    
    metadata : List[int] = []
    metadata.append(min_LOD)
    metadata.append(len(full_shape))
    metadata.append(full_shape[0])
    metadata.append(full_shape[1])
    for i in range(2,len(full_shape)):
        metadata.append(int(full_shape[i]/(2**min_LOD)))
    for i in range(len(nodes)):
        metadata.append(nodes[i].depth)
        metadata.append(nodes[i].index)
        metadata.append(nodes[i].LOD)

    metadata = np.array(metadata, dtype=int)
    metadata.tofile(os.path.join(temp_folder_path, "metadata"))

    os.system("tar -cjvf " + save_location + " -C " + folder + " Temp")
    os.system("rm -r " + temp_folder_path)

def fpzip_compress_nodelist(nodes: OctreeNodeList, full_shape, max_LOD,
downscaling_technique, device, mode,
folder : str, name : str):
    
    nearest_upscale = UpscalingMethod("nearest", device)
    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

    full_im = nodes_to_full_img(nodes, full_shape, max_LOD, nearest_upscale,
    downscaling_technique, device, data_levels, mask_levels,
    data_downscaled_levels, mask_downscaled_levels, mode)
    min_LOD = max_LOD
    for i in range(len(nodes)):
        min_LOD = min(nodes[i].LOD, min_LOD)
    if(min_LOD > 0):
        if(mode == "2D"):
            full_im = subsample_downscale2D(full_im, int(2**min_LOD))
        elif(mode == "3D"):
            full_im = subsample_downscale3D(full_im, int(2**min_LOD))
    temp_folder_path = os.path.join(folder, "Temp")
    save_location = os.path.join(folder, name +".tar.gz")
    if(not os.path.exists(temp_folder_path)):
        os.makedirs(temp_folder_path)
    

    d = full_im.cpu().numpy()[0,0]
    d_loc = os.path.join(temp_folder_path,"nn_data.dat")
    ndims = len(d.shape)
    print(d.shape)
    d.tofile(d_loc)
    command = "fpzip -t float -i " + d_loc + " -o " +\
        d_loc + ".fpzip" + " -" + str(ndims) + " " + \
        str(d.shape[0]) + " " + str(d.shape[1])
    if(ndims == 3):
        command = command + " " + str(d.shape[2])
    print(command)
    os.system(command)
    os.system("rm " + d_loc)
    
    metadata : List[int] = []
    metadata.append(min_LOD)
    metadata.append(len(full_shape))
    metadata.append(full_shape[0])
    metadata.append(full_shape[1])
    for i in range(2,len(full_shape)):
        metadata.append(int(full_shape[i]/(2**min_LOD)))
    for i in range(len(nodes)):
        metadata.append(nodes[i].depth)
        metadata.append(nodes[i].index)
        metadata.append(nodes[i].LOD)

    metadata = np.array(metadata, dtype=int)
    metadata.tofile(os.path.join(temp_folder_path, "metadata"))

    os.system("tar -cjvf " + save_location + " -C " + folder + " Temp")
    os.system("rm -r " + temp_folder_path)

def sz_compress(nodes: OctreeNodeList, full_shape, max_LOD,
downscaling_technique, device, mode,
folder : str, name : str, metric : str, value : float):
    
    
    min_LOD = max_LOD
    for i in range(len(nodes)):
        min_LOD = min(nodes[i].LOD, min_LOD)
    
    if(mode == "2D"):
        full_im = torch.zeros([1, full_shape[1], 
        int(full_shape[2] / (2**min_LOD)), int(full_shape[3] / (2**min_LOD))], dtype=torch.float32)
    elif(mode == "3D"):
        full_im = torch.zeros([1, full_shape[1], 
        int(full_shape[2] / (2**min_LOD)), 
        int(full_shape[3] / (2**min_LOD)),
        int(full_shape[4] / (2**min_LOD))], dtype=torch.float32)

    full_im += nodes.mean().numpy()

    for i in range(len(nodes)):
        n = nodes[i]
        if(mode == "2D"):
            x, y = get_location2D(full_shape[2], full_shape[3], 
            n.depth, n.index)
            x = int(x / (2**min_LOD))
            y = int(y / (2**min_LOD))
            width = n.data.shape[2] 
            height = n.data.shape[3]
            full_im[:,:,x:x+width,y:y+height] = n.data

        elif(mode == "3D"):
            x, y, z = get_location3D(full_shape[2], full_shape[3], full_shape[4],
            n.depth, n.index)
            x = int(x / (2**min_LOD))
            y = int(y / (2**min_LOD))
            z = int(z / (2**min_LOD))
            width = n.data.shape[2] 
            height = n.data.shape[3]
            d = n.data.shape[4]
            full_im[:,:,x:x+width,y:y+height,z:z+d] = n.data
            
    temp_folder_path = os.path.join(folder, "Temp")
    save_location = os.path.join(folder, name +".tar.gz")
    if(not os.path.exists(temp_folder_path)):
        os.makedirs(temp_folder_path)
    if(mode == "2D"):
        imageio.imwrite(os.path.join(folder,name+"compressed.png"), np.transpose(full_im[0], (1, 2, 0)))
    for i in range(full_im.shape[1]):
        d = full_im.cpu().numpy()[0,i]
        d_loc = os.path.join(temp_folder_path,"nn_data_"+str(i)+".dat")
        ndims = len(d.shape)
        print(d.shape)
        d.tofile(d_loc)
        command = "sz -z -f -i " + d_loc + " -" + str(ndims) + " " + \
            str(d.shape[0]) + " " + str(d.shape[1])
        if(ndims == 3):
            command = command + " " + str(d.shape[2])
        '''
        if(min_LOD == 0):
            if(metric == "psnr"):
                command = command + " -M PSNR -S " + str(value)
            elif(metric == "mre"):
                command = command + " -M REL -R " + str(value)
            elif(metric == "pw_mre"):
                command = command + " -M PW_REL -P " + str(value)
        else:
        '''
        #command = command + " -M PW_REL -P 0.001"
        max_diff = nodes.max() - nodes.min()
        REL_target = (max_diff / (10**(value/10))) **0.5
        #REL_target *= 75
        #command = command + " -M REL -R " + str(REL_target)
        #command = command + " -M REL -R 0.0025"
        command = command + " -M PW_REL -P " + str(REL_target)
        print(command)
        os.system(command)
        os.system("rm " + d_loc)
    
    del full_im

    metadata : List[int] = []
    metadata.append(min_LOD)
    metadata.append(len(full_shape))
    metadata.append(full_shape[0])
    metadata.append(full_shape[1])
    for i in range(2,len(full_shape)):
        metadata.append(int(full_shape[i]/(2**min_LOD)))
    for i in range(len(nodes)):
        metadata.append(nodes[i].depth)
        metadata.append(nodes[i].index)
        metadata.append(nodes[i].LOD)

    metadata = np.array(metadata, dtype=int)
    metadata.tofile(os.path.join(temp_folder_path, "metadata"))

    os.system("tar -cjvf " + save_location + " -C " + folder + " Temp")
    os.system("rm -r " + temp_folder_path)

def sz_decompress(filename : str, device : str):
    print("Decompressing " + filename + " with sz method")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(folder_path, "Temp")
    if(not os.path.exists(temp_folder)):
        os.makedirs(temp_folder)
    
    nodes = OctreeNodeList()

    os.system("tar -xvf " + filename)
    metadata = np.fromfile(os.path.join(temp_folder, "metadata"), dtype=int)
    min_LOD = metadata[0]
    full_shape = []
    for i in range(2, metadata[1]+2):
        full_shape.append(metadata[i])
        
    metadata = metadata[metadata[1]+2:]
    
    data_channels = []
    for i in range(full_shape[1]):
        command = "sz -x -f -s " + os.path.join(temp_folder, "nn_data_"+str(i)+".dat.sz") + " -" + \
            str(len(full_shape[2:])) + " " + str(full_shape[2]) + " " + str(full_shape[3])

        if(len(full_shape) == 5):
            command = command + " " + str(full_shape[4])
        print(command)
        os.system(command)

        full_data = np.fromfile(os.path.join(temp_folder, "nn_data_"+str(i)+".dat.sz.out"), 
        dtype=np.float32)
        full_data = np.reshape(full_data, full_shape[2:])
        
        full_data = torch.Tensor(full_data)
        data_channels.append(full_data)

    #print(full_shape)
    full_data = torch.stack(data_channels).unsqueeze(0)
    for i in range(0, len(metadata), 3):
        depth = metadata[i]
        index = metadata[i+1]
        lod = metadata[i+2]
        if(len(full_shape) == 4):
            x, y = get_location2D(int(full_shape[2]*(2**min_LOD)), int(full_shape[3]*(2**min_LOD)), 
            depth, index)
            x = int(x / (2**min_LOD))
            y = int(y / (2**min_LOD))
            width = int(full_shape[2] / (2**(depth+lod-min_LOD)))
            height = int(full_shape[3] / (2**(depth+lod-min_LOD)))
            data = full_data[:,:,x:x+width,y:y+height]

        elif(len(full_shape) == 5):
            x, y, z = get_location3D(int(full_shape[2]*(2**min_LOD)), 
            int(full_shape[3]*(2**min_LOD)), int(full_shape[4]*(2**min_LOD)), 
            depth, index)
            x = int(x / (2**min_LOD))
            y = int(y / (2**min_LOD))
            z = int(z / (2**min_LOD))
            width = int(full_shape[2] / (2**(depth+lod-min_LOD)))
            height = int(full_shape[3] / (2**(depth+lod-min_LOD)))
            d = int(full_shape[4] / (2**(depth+lod-min_LOD)))
            data = full_data[:,:,x:x+width,y:y+height,z:z+d]
        
        #print("Node %i: depth %i index %i lod %i, data shape %s" % (i, depth, index, lod, str(data.shape)))
        #print(str(x) + " " + str(y))
        n = OctreeNode(data.to(device), lod, depth, index)
        nodes.append(n)
    del full_data
    os.system("rm -r " + temp_folder)
    print("Finished decompressing, " + str(len(nodes)) + " blocks recovered")
    return nodes

def sz_decompress_nodelist(filename : str, device : str):
    print("Decompressing " + filename + " with sz method")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(folder_path, "Temp")
    if(not os.path.exists(temp_folder)):
        os.makedirs(temp_folder)
    
    nodes = OctreeNodeList()

    os.system("tar -xvf " + filename)
    metadata = np.fromfile(os.path.join(temp_folder, "metadata"), dtype=int)
    min_LOD = metadata[0]
    full_shape = []
    for i in range(2, metadata[1]+2):
        full_shape.append(metadata[i])
        
    metadata = metadata[metadata[1]+2:]
    
    data_channels = []
    for i in range(full_shape[1]):
        command = "sz -x -f -s " + os.path.join(temp_folder, "nn_data_"+str(i)+".dat.sz") + " -" + \
            str(len(full_shape[2:])) + " " + str(full_shape[2]) + " " + str(full_shape[3])

        if(len(full_shape) == 5):
            command = command + " " + str(full_shape[4])
        print(command)
        os.system(command)

        full_data = np.fromfile(os.path.join(temp_folder, "nn_data_"+str(i)+".dat.sz.out"), 
        dtype=np.float32)
        full_data = np.reshape(full_data, full_shape[2:])
        
        full_data = torch.Tensor(full_data)
        data_channels.append(full_data)

    full_data = torch.stack(data_channels).unsqueeze(0)
    nearest_upscale = UpscalingMethod("nearest", device)

    full_data = nearest_upscale(full_data, int(2**min_LOD))
    for i in range(0, len(metadata), 3):
        depth = metadata[i]
        index = metadata[i+1]
        lod = metadata[i+2]
        if(len(full_shape) == 4):
            x, y = get_location2D(full_data.shape[2], full_data.shape[3], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height]
            data = avgpool_downscale2D(data, int(2**lod))

        elif(len(full_shape) == 5):
            x, y, z = get_location3D(full_data.shape[2], full_data.shape[3], full_data.shape[4], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            d = int(full_data.shape[4] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height,z:z+d]
            data = avgpool_downscale3D(data, int(2**lod))
        
        n = OctreeNode(data.to(device), lod, depth, index)
        nodes.append(n)
    del full_data
    os.system("rm -r " + temp_folder)
    print("Finished decompressing, " + str(len(nodes)) + " blocks recovered")
    return nodes

def zfp_decompress_nodelist(filename : str, device : str):
    print("Decompressing " + filename + " with zfp method")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(folder_path, "Temp")
    if(not os.path.exists(temp_folder)):
        os.makedirs(temp_folder)
    
    nodes = OctreeNodeList()

    os.system("tar -xvf " + filename)
    metadata = np.fromfile(os.path.join(temp_folder, "metadata"), dtype=int)
    min_LOD = metadata[0]
    full_shape = []
    for i in range(2, metadata[1]+2):
        full_shape.append(metadata[i])
        
    metadata = metadata[metadata[1]+2:]
    command = "zfp -f -R -z " + os.path.join(temp_folder, "nn_data.dat.zfp") + " -o " + \
        os.path.join(temp_folder, "nn_data.dat.zfp.out")+ " -" + \
        str(len(full_shape[2:])) + " " + str(full_shape[2]) + " " + str(full_shape[3])

    if(len(full_shape) == 5):
        command = command + " " + str(full_shape[4])
    print(command)
    os.system(command)

    full_data = np.fromfile(os.path.join(temp_folder, "nn_data.dat.zfp.out"), dtype=np.float32)
    print(full_data.shape)
    full_data = np.reshape(full_data, full_shape[2:])
    print(full_data.shape)
    full_data = torch.Tensor(full_data).unsqueeze(0).unsqueeze(0)
    nearest_upscale = UpscalingMethod("nearest", device)

    full_data = nearest_upscale(full_data, int(2**min_LOD))
    print(full_data.shape)
    for i in range(0, len(metadata), 3):
        depth = metadata[i]
        index = metadata[i+1]
        lod = metadata[i+2]
        if(len(full_shape) == 4):
            x, y = get_location2D(full_data.shape[2], full_data.shape[3], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height]
            data = avgpool_downscale2D(data, int(2**lod))

        elif(len(full_shape) == 5):
            x, y, z = get_location3D(full_data.shape[2], full_data.shape[3], full_data.shape[4], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            d = int(full_data.shape[4] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height,z:z+d]
            data = avgpool_downscale3D(data, int(2**lod))
        
        n = OctreeNode(data.to(device), lod, depth, index)
        nodes.append(n)
    os.system("rm -r " + temp_folder)
    print("Finished decompressing, " + str(len(nodes)) + " blocks recovered")
    return nodes

def fpzip_decompress_nodelist(filename : str, device : str):
    print("Decompressing " + filename + " with fpzip method")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(folder_path, "Temp")
    if(not os.path.exists(temp_folder)):
        os.makedirs(temp_folder)
    
    nodes = OctreeNodeList()

    os.system("tar -xvf " + filename)
    metadata = np.fromfile(os.path.join(temp_folder, "metadata"), dtype=int)
    min_LOD = metadata[0]
    full_shape = []
    for i in range(2, metadata[1]+2):
        full_shape.append(metadata[i])
        
    metadata = metadata[metadata[1]+2:]
    command = "fpzip -d -t float -i " + os.path.join(temp_folder, "nn_data.dat.fpzip") + " -o " + \
        os.path.join(temp_folder, "nn_data.dat.fpzip.out")+ " -" + \
        str(len(full_shape[2:])) + " " + str(full_shape[2]) + " " + str(full_shape[3])

    if(len(full_shape) == 5):
        command = command + " " + str(full_shape[4])
    print(command)
    os.system(command)

    full_data = np.fromfile(os.path.join(temp_folder, "nn_data.dat.fpzip.out"), dtype=np.float32)
    full_data = np.reshape(full_data, full_shape[2:])
    
    full_data = torch.Tensor(full_data).unsqueeze(0).unsqueeze(0)
    nearest_upscale = UpscalingMethod("nearest", device)

    full_data = nearest_upscale(full_data, int(2**min_LOD))
    for i in range(0, len(metadata), 3):
        depth = metadata[i]
        index = metadata[i+1]
        lod = metadata[i+2]
        if(len(full_shape) == 4):
            x, y = get_location2D(full_data.shape[2], full_data.shape[3], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height]
            data = avgpool_downscale2D(data, int(2**lod))

        elif(len(full_shape) == 5):
            x, y, z = get_location3D(full_data.shape[2], full_data.shape[3], full_data.shape[4], 
            depth, index)
            width = int(full_data.shape[2] / (2**depth))
            height = int(full_data.shape[3] / (2**depth))
            d = int(full_data.shape[4] / (2**depth))
            data = full_data[:,:,x:x+width,y:y+height,z:z+d]
            data = avgpool_downscale3D(data, int(2**lod))
        
        n = OctreeNode(data.to(device), lod, depth, index)
        nodes.append(n)
    os.system("rm -r " + temp_folder)
    print("Finished decompressing, " + str(len(nodes)) + " blocks recovered")
    return nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--folder',default="octree_files",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--output_folder',default="mag2D_4010",type=str,help='Where to save results')
    parser.add_argument('--save_name',default="Temp",type=str,help='Name for trial in results.pkl')
    parser.add_argument('--start_metric',default=10,type=float,help='PSNR to start tests at')
    parser.add_argument('--end_metric',default=100,type=float,help='PSNR to end tests at')
    parser.add_argument('--metric_skip',default=10,type=float,help='PSNR increment by')
    
    parser.add_argument('--max_LOD',default=6,type=int)
    parser.add_argument('--min_chunk',default=16,type=int)
    parser.add_argument('--device',default="cuda",type=str)
    parser.add_argument('--upscaling_technique',default="bicubic",type=str)
    parser.add_argument('--downscaling_technique',default="avgpool2D",type=str)
    parser.add_argument('--criterion',default="psnr",type=str)
    parser.add_argument('--load_existing',default="false",type=str2bool)
    parser.add_argument('--mode',default="2D",type=str)
    parser.add_argument('--model_name',default="SSR_isomag2D",type=str)
    parser.add_argument('--debug',default="false",type=str2bool)
    parser.add_argument('--distributed',default="false",type=str2bool)
    parser.add_argument('--use_compressor',default="true",type=str2bool)
    parser.add_argument('--save_netcdf',default="false",type=str2bool)
    parser.add_argument('--save_netcdf_octree',default="false",type=str2bool)
    parser.add_argument('--save_TKE',default="false",type=str2bool)

    parser.add_argument('--compressor',default="zfp",type=str)
    parser.add_argument('--data_type',default="h5",type=str)

    parser.add_argument('--dynamic_downscaling',default="true",type=str2bool)
    parser.add_argument('--interpolation_heuristic',default="false",type=str2bool)
    parser.add_argument('--preupscaling_PSNR', type=str2bool, default="false")
    torch.set_grad_enabled(False)

    args = vars(parser.parse_args())
    if(not args['use_compressor']):
        args['compressor'] = "none"
    max_LOD : int = args['max_LOD']
    min_chunk : int = args['min_chunk']
    device: str = args['device']
    upscaling_technique : str = args['upscaling_technique']
    downscaling_technique : str = args['downscaling_technique']
    criterion : str = args['criterion']
    load_existing : bool = args['load_existing']
    mode : str = args['mode']
    model_name : str = args['model_name']
    debug : bool = args['debug']
    distributed : bool = args['distributed']

    img_name : str = args['file'].split(".")[0]
    img_ext : str = args['file'].split(".")[1]
    img_type : str = args['file'].split(".")[1]

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
    results['compression_time'] = []
    results['num_nodes'] = []
    results['rec_psnr'] = []
    results['rec_ssim'] = []
    results['rec_mre'] = []
    results['rec_pwmre'] = []
    results['rec_inner_mre'] = []
    results['rec_inner_pwmre'] = []
    if(args['save_TKE']):
        results['TKE_error'] = []

    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
        if(args['save_name'] in all_data.keys()):
            results = all_data[args['save_name']]
        else:
            all_data[args['save_name']] = results
    else:
        all_data = {}
        all_data[args['save_name']] = results

    if(args['data_type'] == "image"):
        img_gt : torch.Tensor = torch.from_numpy(imageio.imread(
            "TestingData/quadtree_images/"+img_name+"."+img_ext).astype(np.float32)).to(device)
        img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
    elif(args['data_type'] == "h5"):
        f = h5py.File(os.path.join(input_folder, args['file']), "r")
        img_gt : torch.Tensor = torch.from_numpy(np.array(f['data'])).unsqueeze(0).to(device)
        f.close()
    imageio.imwrite(os.path.join(save_folder, "GT.png"), 
        to_img(img_gt, mode))

    full_shape : List[int] = list(img_gt.shape)
    print(full_shape)
    m = args['start_metric']
    upscaling : UpscalingMethod = UpscalingMethod(upscaling_technique, device, 
            model_name, distributed)
    while(m < args['end_metric']):
        torch.cuda.empty_cache()
        #with LineProfiler(mixedLOD_octree_SR_compress,nodes_to_full_img,compress_nodelist) as prof:
        criterion_value = m
        save_name = args['save_name'] + "_"+ criterion + str(m)
        compress_time = 0

        upscaling.change_method(upscaling_technique)
        if not load_existing:
            root_node = OctreeNode(img_gt, 0, 0, 0)
            nodes : OctreeNodeList = OctreeNodeList()
            nodes.append(root_node)
            #torch.save(nodes, './Output/'+img_name+'.torch')
            #nodelist_to_h5(nodes, './Output/'+img_name+'.h5')
            ##############################################
            #nodes : OctreeNodeList = torch.load('./Output/'+img_name+'.torch')
            if(args['interpolation_heuristic']):
                if(args['mode'] == '2D'):
                    upscaling.change_method("bilinear")
                elif(args['mode'] == '3D'):
                    upscaling.change_method("trilinear")   
            start_time : float = time.time()
            if(args['dynamic_downscaling']):
                nodes : OctreeNodeList = mixedLOD_octree_SR_compress(
                    nodes, img_gt, criterion, criterion_value,
                    upscaling, downscaling_technique,
                    min_chunk, max_LOD, device, mode)
                end_time : float = time.time()
                compress_time = end_time - start_time
                #print("Compression took %s seconds" % (str(end_time - start_time)))
                
                num_nodes : int = len(nodes)
                nodes = compress_nodelist(nodes, full_shape, min_chunk, device, mode)
                concat_num_nodes : int = len(nodes)

                print("Concatenating blocks turned %s blocks into %s" % (str(num_nodes), str(concat_num_nodes)))
            else:
                nodes : OctreeNodeList = nondynamic_SR_compress(
                    nodes, img_gt, criterion, criterion_value,
                    upscaling, downscaling_technique,
                    min_chunk, max_LOD, device, mode)
                end_time : float = time.time()
                compress_time = end_time - start_time            
                num_nodes : int = len(nodes)
            print("Compression time %f" % compress_time)
            if(args['preupscaling_PSNR']):
                upscaling.change_method(upscaling_technique)
                data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
                    create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

                img_upscaled = nodes_to_full_img(nodes, full_shape, 
                max_LOD, upscaling, 
                downscaling_technique, device, data_levels, 
                mask_levels, data_downscaled_levels, 
                mask_downscaled_levels, mode)
                while(len(data_levels)>0):
                    del data_levels[0]
                    del mask_levels[0]
                    del data_downscaled_levels[0]
                    del mask_downscaled_levels[0]
                p : float = PSNR(img_upscaled, img_gt).cpu().item()
                print("Pre compression PSNR: " + str(p))
                del img_upscaled
            else:
                p = m
            if(args['use_compressor']):
                if(args['compressor'] == "sz"):
                    sz_compress(nodes, full_shape, max_LOD,
                        downscaling_technique, device, mode,
                        save_folder, save_name,
                        criterion, p)
                elif(args['compressor'] == "zfp"):
                    zfp_compress_nodelist(nodes, full_shape, max_LOD,
                        downscaling_technique, device, mode,
                        save_folder, save_name)
                elif(args['compressor'] == "fpzip"):
                    fpzip_compress_nodelist(nodes, full_shape, max_LOD,
                        downscaling_technique, device, mode,
                        save_folder, save_name)
            else:
                torch.save(nodes, os.path.join(save_folder,
                    save_name+".torch"))
            
            
            #nodelist_to_h5(nodes, "./Output/"+img_name+"_"+upscaling_technique+ \
            #    "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            #        "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+".h5")
        if(args['use_compressor']):
            if(args['compressor'] == "sz"):                
                nodes = sz_decompress(os.path.join(save_folder,save_name + ".tar.gz"), device)
            elif(args['compressor'] == "zfp"):                
                nodes = zfp_decompress_nodelist(os.path.join(save_folder,save_name + ".tar.gz"), device)
            elif(args['compressor'] == "fpzip"):
                nodes = fpzip_decompress_nodelist(os.path.join(save_folder,save_name + ".tar.gz"), device)
        else:
            nodes : OctreeNodeList = torch.load(os.path.join(save_folder,
                save_name+".torch"))

        upscaling.change_method(upscaling_technique)
        data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

        t01 = time.time()
        img_upscaled = nodes_to_full_img(nodes, full_shape, 
        max_LOD, upscaling, 
        downscaling_technique, device, data_levels, 
        mask_levels, data_downscaled_levels, 
        mask_downscaled_levels, mode)
        print("Recon time %f" % (time.time() - t01))
        while(len(data_levels)>0):
            del data_levels[0]
            del mask_levels[0]
            del data_downscaled_levels[0]
            del mask_downscaled_levels[0]
        imageio.imwrite(os.path.join(save_folder, save_name+".png"), 
                to_img(img_upscaled, mode))

        if(args['use_compressor']):
            f_size_kb = os.path.getsize(os.path.join(save_folder,
            save_name+".tar.gz")) / 1024
        else:
            f_size_kb = os.path.getsize(os.path.join(save_folder,
                save_name+".torch")) / 1024

        f_data_size_kb = nodes.total_size()
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            final_psnr : float = PSNR(img_upscaled, img_gt).cpu().item()
            final_mse : float = MSE(img_upscaled, img_gt).cpu().item()
            final_mre : float = relative_error(img_upscaled, img_gt).cpu().item()
            final_pwmre: float = pw_relative_error(img_upscaled, img_gt).cpu().item()
            torch.cuda.empty_cache()
            if(len(img_upscaled.shape) == 4):
                final_ssim : float = ssim(img_upscaled, img_gt).cpu().item()
                final_inner_mre : float = relative_error(
                    img_upscaled[:,:,20:img_upscaled.shape[2]-20,20:img_upscaled.shape[3]-20], 
                    img_gt[:,:,20:img_gt.shape[2]-20,20:img_gt.shape[3]-20]).cpu().item()
                final_inner_pwmre: float = pw_relative_error(
                    img_upscaled[:,:,20:img_upscaled.shape[2]-20,20:img_upscaled.shape[3]-20], 
                    img_gt[:,:,20:img_gt.shape[2]-20,20:img_gt.shape[3]-20]).cpu().item()
            elif(len(img_upscaled.shape) == 5):
                final_ssim : float = ssim3D(img_upscaled.detach().to("cuda:1"), img_gt.detach().to("cuda:1")).cpu().item()
                
                final_inner_mre : float = relative_error(
                    img_upscaled[:,:,
                        20:img_upscaled.shape[2]-20,
                        20:img_upscaled.shape[3]-20,
                        20:img_upscaled.shape[4]-20], 
                    img_gt[:,:,
                        20:img_gt.shape[2]-20,
                        20:img_gt.shape[3]-20,
                        20:img_gt.shape[4]-20]).cpu().item()
                final_inner_pwmre: float = pw_relative_error(
                    img_upscaled[:,:,
                        20:img_upscaled.shape[2]-20,
                        20:img_upscaled.shape[3]-20,
                        20:img_upscaled.shape[4]-20], 
                    img_gt[:,:,
                        20:img_gt.shape[2]-20,
                        20:img_gt.shape[3]-20,
                        20:img_gt.shape[4]-20]).cpu().item()
        print(img_upscaled.shape)
        print("Final stats:")
        print("Target - " + criterion + " " + str(criterion_value))
        print("PSNR: %0.02f, SSIM: %0.04f, MSE: %0.02f, MRE: %0.04f, PWMRE: %0.04f" % \
            (final_psnr, final_ssim, final_mse, final_mre, final_pwmre))
        print("Pre-compressed data size: %f kb" % nodes.total_size())
        print("Saved file size: %f kb" % f_size_kb)
        results['psnrs'].append(criterion_value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compress_time)
        results['num_nodes'].append(len(nodes))
        results['rec_psnr'].append(final_psnr)
        results['rec_ssim'].append(final_ssim)
        results['rec_mre'].append(final_mre)
        results['rec_pwmre'].append(final_pwmre)
        results['rec_inner_mre'].append(final_inner_mre)
        results['rec_inner_pwmre'].append(final_inner_pwmre)
        if(args['save_TKE']):
            results['TKE_error'].append(0.5*((img_gt**2).mean().item()-(img_upscaled**2).mean().item()))
        all_data[args['save_name']] = results
        save_obj(all_data, os.path.join(save_folder, "results.pkl"))

        if(args['save_netcdf']):
            from netCDF4 import Dataset
            rootgrp = Dataset(save_name+".nc", "w", format="NETCDF4")
            rootgrp.createDimension("u")
            rootgrp.createDimension("v")
            if(mode == "3D"):
                rootgrp.createDimension("w")
            rootgrp.createDimension("channels", img_upscaled.shape[1])
            if(mode == "3D"):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v","w"))
            elif(mode == "2D"):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v"))
            dim_0[:] = img_upscaled[0,0].cpu().numpy()
        del img_upscaled

        if(args['save_netcdf_octree']):

            upscaling.change_method('nearest')

            data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

            img_upscaled_point = nodes_to_full_img(nodes, full_shape, 
            max_LOD, upscaling, 
            downscaling_technique, device, data_levels, 
            mask_levels, data_downscaled_levels, 
            mask_downscaled_levels, mode)
            while(len(data_levels)>0):
                del data_levels[0]
                del mask_levels[0]
                del data_downscaled_levels[0]
                del mask_downscaled_levels[0]

            from netCDF4 import Dataset
            rootgrp = Dataset(save_name+"octree.nc", "w", format="NETCDF4")
            rootgrp.createDimension("u")
            rootgrp.createDimension("v")
            if(mode == "3D"):
                rootgrp.createDimension("w")
            rootgrp.createDimension("channels", img_upscaled_point.shape[1])
            if(mode == "3D"):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v","w"))
            elif(mode == "2D"):
                dim_0 = rootgrp.createVariable("velocity magnitude", np.float32, ("u","v"))
            dim_0[:] = img_upscaled_point[0,0].cpu().numpy()
            del img_upscaled_point

        if(args['debug']):           
            upscaling.change_method(upscaling_technique)
            t01 = time.time()
            img_seams = nodes_to_full_img_seams(nodes, full_shape,
            upscaling, device, mode)
            print("Upscaling with seams time: %f" % (time.time() - t01))
            imageio.imwrite(os.path.join(save_folder, save_name+"_seams.png"), 
                    to_img(img_seams, mode))

            img_upscaled_debug, cmap = nodes_to_full_img_debug(nodes, full_shape, 
            max_LOD, upscaling, 
            downscaling_technique, device, mode)
            img_upscaled_debug = img_upscaled_debug.cpu().numpy().astype(np.uint8)
            if(mode == "3D"):
                img_upscaled_debug = img_upscaled_debug[:,:,:,
                :,int(img_upscaled_debug.shape[4]/2)+1]
            img_upscaled_debug = img_upscaled_debug[0]
            img_upscaled_debug = np.transpose(img_upscaled_debug, (1, 2, 0))
            imageio.imwrite(os.path.join(save_folder, save_name+"_debug.png"), 
                    img_upscaled_debug)
            del img_upscaled_debug
            imageio.imwrite(os.path.join(save_folder,"colormap.png"), 
            cmap.cpu().numpy().astype(np.uint8))

            point_us = "point2D" if mode == "2D" else "point3D"
            upscaling.change_method('nearest')

            data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
            create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)

            img_upscaled_point = nodes_to_full_img(nodes, full_shape, 
            max_LOD, upscaling, 
            downscaling_technique, device, data_levels, 
            mask_levels, data_downscaled_levels, 
            mask_downscaled_levels, mode)
            while(len(data_levels)>0):
                del data_levels[0]
                del mask_levels[0]
                del data_downscaled_levels[0]
                del mask_downscaled_levels[0]
            imageio.imwrite(os.path.join(save_folder, save_name+"_point.png"), 
                    to_img(img_upscaled_point, mode))
            del img_upscaled_point
        for i in range(len(nodes)):
            del nodes[i].data
        del nodes
        #print(prof.display())
        m += args['metric_skip']

    '''
    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
    else:
        all_data = {}
    '''
    

        
