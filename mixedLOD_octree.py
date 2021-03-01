import imageio
import numpy as np
import torch
import torch.nn.functional as F
from math import log2
from utility_functions import *
import time
from typing import Dict, List, Tuple, Optional
import h5py

#@torch.jit.script
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

#@torch.jit.script
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

#@torch.jit.script
def get_location3D(full_height: int, full_width : int, full_depth : int,
depth : int, index : int) -> Tuple[int, int]:
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

#@torch.jit.script
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

def ssim_criterion(GT_image, img, min_ssim=0.6) -> float:
    return ssim(img.permute(2, 0, 1).unsqueeze(0), GT_image.permute(2, 0, 1).unsqueeze(0)) > min_ssim

#@torch.jit.script
def MSE(x, GT) -> float:
    return ((x-GT)**2).mean()

#@torch.jit.script
def PSNR(x, GT) -> float:
    max_diff : float = GT.max() - GT.min()
    return 20 * torch.log(torch.tensor(max_diff)) - 10*torch.log(MSE(x, GT))

#@torch.jit.script
def relative_error(x, GT) -> float:
    max_diff : float = GT.max() - GT.min()
    return torch.abs(GT-x).max() / max_diff

#@torch.jit.script
def psnr_criterion(GT_image, img, min_PSNR : float) -> bool:
    return PSNR(img, GT_image) > min_PSNR

#@torch.jit.script
def mse_criterion(GT_image, img, max_mse : float) -> bool:
    return MSE(img, GT_image) < max_mse

#@torch.jit.script
def maximum_relative_error(GT_image, img, max_e : float) -> bool:
    return relative_error(img, GT_image) < max_e

#@torch.jit.script
def bilinear_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    align_corners=False, mode='bilinear')
    return img

#@torch.jit.script
def trilinear_upscale(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = F.interpolate(vol, scale_factor=float(scale_factor), 
    align_corners=False, mode='trilinear')
    return vol

#@torch.jit.script
def bicubic_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    align_corners=False, mode='bicubic')
    img = img.clamp_(0.0, 255.0)
    return img

#@torch.jit.script
def point_upscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    upscaled_img = torch.zeros([img.shape[0], img.shape[1],
    int(img.shape[2]*scale_factor), 
    int(img.shape[3]*scale_factor)]).to(img.device)
    
    for x in range(img.shape[2]):
        for y in range(img.shape[3]):
            upscaled_img[:,:,x*scale_factor:(x+1)*scale_factor, 
            y*scale_factor:(y+1)*scale_factor] = img[:,:,x,y].view(img.shape[0], img.shape[1], 1, 1).repeat(1, 1, scale_factor, scale_factor)
    return upscaled_img

#@torch.jit.script
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

#@torch.jit.script
def nearest_neighbor_upscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=float(scale_factor), 
    mode='nearest')
    return img

#@torch.jit.script
def bilinear_downscale(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = F.interpolate(img, scale_factor=(1/scale_factor), align_corners=True, mode='bilinear')
    return img

#@torch.jit.script
def avgpool_downscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = AvgPool2D(img, scale_factor)
    return img

#@torch.jit.script
def avgpool_downscale3D(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = AvgPool3D(vol, scale_factor)
    return vol

#@torch.jit.script
def subsample_downscale2D(img : torch.Tensor, scale_factor : int) -> torch.Tensor:
    img = img[:,:, ::2, ::2]
    return img

#@torch.jit.script
def subsample_downscale3D(vol : torch.Tensor, scale_factor : int) -> torch.Tensor:
    vol = vol[:,:, ::2, ::2, ::2]
    return vol

#@torch.jit.script
def upscale(method: str, img: torch.Tensor, scale_factor: int) -> torch.Tensor:
    up = torch.zeros([1])
    if(method == "bilinear"):
        up = bilinear_upscale(img, scale_factor)
    elif(method == "bicubic"):
        up = bicubic_upscale(img, scale_factor)
    elif(method == "point2D"):
        up = point_upscale2D(img, scale_factor)
    elif(method == "point3D"):
        up = point_upscale3D(img, scale_factor)
    elif(method == "nearest"):
        up = nearest_neighbor_upscale(img, scale_factor)
    elif(method == "trilinear"):
        up = trilinear_upscale(img, scale_factor)
    else:
        print("No support for upscaling method: " + str(method))
    return up

#@torch.jit.script
def downscale(method: str, img: torch.Tensor, scale_factor: int) -> torch.Tensor:
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

#@torch.jit.script
def criterion_met(method: str, value: float, 
a: torch.Tensor, b: torch.Tensor) -> bool:
    passed = False
    if(method == "psnr"):
        passed = psnr_criterion(a, b, value)
    elif(method == "mse"):
        passed = mse_criterion(a, b, value)
    elif(method == "mre"):
        passed = maximum_relative_error(a, b, value)
    else:
        print("No support for criterion: " + str(method))
    return passed

#@torch.jit.script
def nodes_to_downscaled_levels(nodes : OctreeNodeList, full_shape : List[int],
    max_LOD : int, downscaling_technique: str, device : str, 
    data_levels: List[torch.Tensor], mask_levels:List[torch.Tensor],
    data_downscaled_levels: List[torch.Tensor], mask_downscaled_levels:List[torch.Tensor],
    mode : str):
    

    i : int = len(data_downscaled_levels) - 2
    mask_downscaled_levels[-1][:] = mask_levels[-1][:]
    data_downscaled_levels[-1][:] = data_levels[-1][:]

    #imageio.imwrite("data_"+str(i+1)+"_filled_in.png", data_downscaled_levels[-1].cpu().numpy().astype(np.uint8)) 
    #imageio.imwrite("mask_"+str(i+1)+"_filled_in.png", mask_downscaled_levels[-1].cpu().numpy().astype(np.uint8)*255)
    while i >= 0:
        data_down = downscale(downscaling_technique, 
        data_downscaled_levels[i+1], 2)
        if(mode == "3D"):
            mask_down = mask_downscaled_levels[i+1][:,:,::2,::2,::2]
        elif(mode == "2D"):
            mask_down = mask_downscaled_levels[i+1][:,:,::2,::2]
        
        #imageio.imwrite("data_"+str(i)+"_downscaled.png", data_down.cpu().numpy().astype(np.uint8))        
        #imageio.imwrite("mask_"+str(i)+"_downscaled.png", mask_down.cpu().numpy().astype(np.uint8)*255)

        data_downscaled_levels[i] = data_down + data_levels[i]
        mask_downscaled_levels[i] = mask_down + mask_levels[i]
        
        #imageio.imwrite("data_"+str(i)+"_filldedin.png", data_downscaled_levels[i].cpu().numpy().astype(np.uint8))        
        #imageio.imwrite("mask_"+str(i)+"_filledin.png", mask_downscaled_levels[i].cpu().numpy().astype(np.uint8)*255)

        i -= 1
        
#@torch.jit.script
def nodes_to_full_img(nodes: OctreeNodeList, full_shape: List[int], 
    max_LOD : int, upscaling_technique : str, 
    downscaling_technique : str, device : str, 
    data_levels: List[torch.Tensor], mask_levels:List[torch.Tensor],
    data_downscaled_levels: List[torch.Tensor], mask_downscaled_levels:List[torch.Tensor],
    mode : str) -> torch.Tensor:

    nodes_to_downscaled_levels(nodes, 
    full_shape, max_LOD, downscaling_technique,
    device, data_levels, mask_levels, data_downscaled_levels, 
    mask_downscaled_levels, mode)
    
    curr_LOD = max_LOD
    full_img = data_downscaled_levels[0]
    
    i = 0
    while(curr_LOD > 0):
        
        full_img = upscale(upscaling_technique, full_img, 2)
        curr_LOD -= 1
        i += 1

        full_img = full_img * (1-mask_downscaled_levels[i]) + \
             data_downscaled_levels[i]*mask_downscaled_levels[i]
    return full_img

#@torch.jit.script
def nodes_to_full_img_debug(nodes: OctreeNodeList, full_shape: List[int], 
max_LOD : int, upscaling_technique : str, 
downscaling_technique : str, device : str, mode : str) -> Tuple[torch.Tensor, torch.Tensor]:
    full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3]]).to(device)
    if(mode == "3D"):
        full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3], full_shape[4]]).to(device)
    cmap : List[torch.Tensor] = [
        torch.tensor([[255, 255, 255]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[247, 251, 162]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[244, 189, 55]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[233, 112, 37]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[178, 52, 85]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[115, 30, 107]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[37, 15, 77]], dtype=nodes[0].data.dtype, device=device),
        torch.tensor([[0, 0, 0]], dtype=nodes[0].data.dtype, device=device)
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
                    int((curr_node.data.shape[4]*(2**curr_node.LOD))),
            ] = torch.zeros([full_shape[0], 3, 
            curr_node.data.shape[2]*(2**curr_node.LOD),
            curr_node.data.shape[3]*(2**curr_node.LOD),
            curr_node.data.shape[4]*(2**curr_node.LOD)])
            full_img[
                int(x_start)+1: \
                int(x_start)+ \
                    int((curr_node.data.shape[2]*(2**curr_node.LOD)))-1,
                int(y_start)+1: \
                int(y_start)+ \
                    int((curr_node.data.shape[3]*(2**curr_node.LOD)))-1,
                int(z_start)+1: \
                int(z_start)+ \
                    int((curr_node.data.shape[4]*(2**curr_node.LOD)))-1,

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

#@torch.jit.script
def nodes_to_full_img_seams(nodes: OctreeNodeList, full_shape: List[int], 
upscaling_technique : str, device: str, mode : str):
    full_img = torch.zeros(full_shape).to(device)
    
    # 1. Fill in known data
    for i in range(len(nodes)):
        curr_node = nodes[i]
        if(mode == "2D"):
            x_start, y_start = get_location2D(full_shape[2], full_shape[3], curr_node.depth, curr_node.index)
            img_part = upscale(upscaling_technique, curr_node.data, (2**curr_node.LOD))
            full_img[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3]] = img_part
        elif(mode == "3D"):
            x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], curr_node.depth, curr_node.index)
            img_part = upscale(upscaling_technique, curr_node.data, (2**curr_node.LOD))
            full_img[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3],z_start:z_start+img_part.shape[4]] = img_part
    
    return full_img

#@torch.jit.script
def remove_node_from_data_caches(node: OctreeNode, full_shape: List[int],
data_levels: List[torch.Tensor], mask_levels: List[torch.Tensor], mode : str):

    curr_ds_ratio = (2**node.LOD)
    if(mode == "2D"):
        x_start, y_start = get_location2D(full_shape[2], full_shape[3], node.depth, node.index)
        ind = len(data_levels) - 1 - int(torch.log2(torch.tensor(float(curr_ds_ratio))).item())
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
        ind = len(data_levels) - 1 - int(torch.log2(torch.tensor(float(curr_ds_ratio))).item())
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

#@torch.jit.script
def add_node_to_data_caches(node: OctreeNode, full_shape: List[int],
data_levels: List[torch.Tensor], mask_levels: List[torch.Tensor], mode : str):
    curr_ds_ratio = (2**node.LOD)
    if(mode == "2D"):
        x_start, y_start = get_location2D(full_shape[2], full_shape[3], node.depth, node.index)
        ind = len(data_levels) - node.LOD - 1
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
        ind = len(data_levels) - node.LOD - 1
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

#@torch.jit.script
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
        curr_shape : List[int] = [full_shape[0], full_shape[1], full_shape[2], full_shape[3], full_shape[4]]
    while(curr_LOD <= max_LOD):
        full_img = torch.zeros(curr_shape).to(device)
        mask = torch.zeros(curr_shape).to(device)
        data_levels.insert(0, full_img.clone())
        data_downscaled_levels.insert(0, full_img.clone())
        mask_levels.insert(0, mask.clone())
        mask_downscaled_levels.insert(0, mask.clone())
        curr_shape[2] = int(curr_shape[2] / 2)
        curr_shape[3] = int(curr_shape[3] / 2)  
        if(mode == "3D"):
            curr_shape[4] = int(curr_shape[4] / 2)
        curr_LOD += 1
    
    for i in range(len(nodes)):
        add_node_to_data_caches(nodes[i], full_shape,
        data_levels, mask_levels, mode)
    
    return data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels

#@torch.jit.script
def mixedLOD_octree_SR_compress(
    nodes : OctreeNodeList, GT_image : torch.Tensor, 
    criterion: str, criterion_value : float,
    upscaling_technique: str, downscaling_technique: str,
    min_chunk_size : int, max_LOD : int, 
    device : str, mode : str
    ) -> OctreeNodeList:
    node_indices_to_check = [ 0 ]
    nodes_checked = 0
    full_shape = nodes[0].data.shape
    
    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
        create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)
    
    add_node_to_data_caches(nodes[0], full_shape, data_levels, mask_levels, mode)

    while(len(node_indices_to_check) > 0): 
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
        upscaling_technique, downscaling_technique,
        device, data_levels, mask_levels, data_downscaled_levels, 
        mask_downscaled_levels, mode)
        # If criterion not met, reset data and stride, and see
        # if the node is large enough to split into subnodes
        # Otherwise, we keep the downsample, and add the node back as a 
        # leaf node
        if(not criterion_met(criterion, criterion_value, GT_image, new_img)):
            remove_node_from_data_caches(n, full_shape, data_levels, mask_levels, mode)
            n.data = original_data
            n.LOD = n.LOD - 1
            
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
        else:
            if(n.LOD < max_LOD and 
                n.min_width()*(2**n.LOD) > min_chunk_size and
                n.min_width() > 1):
                node_indices_to_check.append(i)
    
    print("Nodes traversed: " + str(nodes_checked))
    return nodes            

def compress_nodelist(nodes: OctreeNodeList, full_size : List[int], 
min_chunk_size: int, device : str, mode : str) -> OctreeNodeList:

    min_width : int = full_size[2]
    for i in range(3, len(full_size)):
        min_width = min(min_width, full_size[i])

    current_depth : int = int(torch.log2(torch.tensor(min_width/min_chunk_size)))

    # dict[depth -> LOD -> group parent index -> list]
    groups : Dict[int, Dict[int, Dict[int, Dict[int, OctreeNode]]]] = {}

    magic_num : int = 4 if mode == "2D" else 8
    for i in range(len(nodes)):
        d : int = nodes[i].depth
        l : int = nodes[i].LOD
        group_parent_index : int = int(nodes[i].index / magic_num)
        n_index : int = nodes[i].index % magic_num

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
                            group[0].depth-1, int(group[0].index / 4))
                            nodes.append(new_node)
                            nodes.remove(group[0])
                            nodes.remove(group[1])
                            nodes.remove(group[2])
                            nodes.remove(group[3])
                            d = current_depth-1
                            if(d not in groups.keys()):
                                groups[d] = {}
                            if(lod not in groups[d].keys()):
                                groups[d][l] = {}
                            if(int(parent/4) not in groups[d][l].keys()):
                                groups[d][l][int(parent/4)] = {}
                            groups[d][l][int(parent/4)][new_node.index % 4] = new_node
        current_depth -= 1
    return nodes


def to_img(input : torch.Tensor, mode : str):
    if(mode == "2D"):
        img = input[0].permute(1, 2, 0).cpu().numpy()
        img -= img.min()
        img *= (255/img.max())
        img = img.astype(np.uint8)
    elif(mode == "3D"):
        img = input[0,:,:,:,int(input.shape[4]/2)].permute(1, 2, 0).cpu().numpy()
        img -= img.min()
        img *= (255/img.max())
        img = img.astype(np.uint8)
    return img

if __name__ == '__main__':
    max_LOD : int = 6
    min_chunk : int = 16
    device: str = "cuda"
    upscaling_technique : str = "bicubic"
    downscaling_technique : str = "avgpool2D"
    criterion : str = "mre"
    criterion_value : float = 0.05
    load_existing = False
    mode : str = "2D"

    img_name : str = "4010"
    img_ext : str = "h5"
    img_type : str = "h5"

    if(img_type == "image"):
        img_gt : torch.Tensor = torch.from_numpy(imageio.imread(
            "TestingData/quadtree_images/"+img_name+"."+img_ext).astype(np.float32)).to(device)
        img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
    elif(img_type == "h5"):
        f = h5py.File("TestingData/quadtree_images/"+img_name+"."+img_ext, 'r')
        img_gt : torch.Tensor = torch.from_numpy(np.array(f['data'])).unsqueeze(0).to(device)
        f.close()

    full_shape : List[int] = list(img_gt.shape)

    if not load_existing:
        root_node = OctreeNode(img_gt, 0, 0, 0)
        nodes : OctreeNodeList = OctreeNodeList()
        nodes.append(root_node)
        torch.save(nodes, './Output/'+img_name+'.torch')

        ##############################################
        nodes : OctreeNodeList = torch.load('./Output/'+img_name+'.torch')
        start_time : float = time.time()
        nodes : OctreeNodeList = mixedLOD_octree_SR_compress(
            nodes, img_gt, criterion, criterion_value,
            upscaling_technique, downscaling_technique,
            min_chunk, max_LOD, device, mode)
            
        end_time : float = time.time()
        print("Compression took %s seconds" % (str(end_time - start_time)))
        

        num_nodes : int = len(nodes)
        nodes = compress_nodelist(nodes, full_shape, min_chunk, device, mode)
        concat_num_nodes : int = len(nodes)

        print("Concatenating blocks turned %s blocks into %s" % (str(num_nodes), str(concat_num_nodes)))

        torch.save(nodes, "./Output/"+img_name+"_"+upscaling_technique+ \
            "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
                "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+".torch")
    

    nodes : OctreeNodeList = torch.load("./Output/"+img_name+"_"+upscaling_technique+ \
        "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+".torch")

    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
        create_caches_from_nodelist(nodes, full_shape, max_LOD, device, mode)


    img_upscaled = nodes_to_full_img(nodes, full_shape, 
    max_LOD, upscaling_technique, 
    downscaling_technique, device, data_levels, 
    mask_levels, data_downscaled_levels, 
    mask_downscaled_levels, mode)

    imageio.imwrite("./Output/"+img_name+"_"+upscaling_technique+ \
        "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+".jpg", 
            to_img(img_upscaled, mode))

            


    img_seams = nodes_to_full_img_seams(nodes, full_shape,
    upscaling_technique, device, mode)

    imageio.imwrite("./Output/"+img_name+"_"+upscaling_technique+ \
        "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+"_seams.jpg", 
            to_img(img_seams, mode))



    img_upscaled_debug, cmap = nodes_to_full_img_debug(nodes, full_shape, 
    max_LOD, upscaling_technique, 
    downscaling_technique, device, mode)

    imageio.imwrite("./Output/"+img_name+"_"+upscaling_technique+ \
        "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+"_debug.jpg", 
            to_img(img_upscaled_debug, mode))

    imageio.imwrite("./Output/colormap.jpg", cmap.cpu().numpy().astype(np.uint8))

    point_us = "point2D" if mode == "2D" else "point3D"
    img_upscaled_point = nodes_to_full_img(nodes, full_shape, 
    max_LOD, point_us, 
    downscaling_technique, device, data_levels, 
    mask_levels, data_downscaled_levels, 
    mask_downscaled_levels, mode)
    
    imageio.imwrite("./Output/"+img_name+"_"+upscaling_technique+ \
        "_"+downscaling_technique+"_"+criterion+str(criterion_value)+"_" +\
            "maxlod"+str(max_LOD)+"_chunk"+str(min_chunk)+"_point.jpg", 
            to_img(img_upscaled_point, mode))



    final_psnr : float = PSNR(img_upscaled, img_gt)
    final_mse : float = MSE(img_upscaled, img_gt)
    final_mre : float = relative_error(img_upscaled, img_gt)

    print("Final stats:")
    print("PSNR: %0.02f, MSE: %0.02f, MRE: %0.04f" % (final_psnr, final_mse, final_mre))
    print("Saved data size: %f kb" % nodes.total_size())
