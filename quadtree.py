import imageio
import numpy as np
import torch
import torch.nn.functional as F
import json
from json import JSONEncoder, JSONDecoder
from utility_functions import *
import time

@torch.jit.script
def MSE(x, GT):
    return ((x-GT)**2).mean()

@torch.jit.script
def PSNR(x, GT):
    max_diff = 255.0
    return 20 * torch.log(torch.tensor(max_diff)) - 10*torch.log(MSE(x, GT))

def upsample_from_quadtree_start(quadtree,device="cuda"):
    full_img = torch.zeros(quadtree['full_shape']).to(device)
    trees = [quadtree]
    while(len(trees) > 0):
        curr_tree = trees.pop(0)
        if(len(curr_tree['children']) == 0):
            # Leaf nodes have data and can be upscaled
            full_img = upsample_from_quadtree(curr_tree, full_img,device=device)
        else:
            # Other nodes have children that have data
            for i in range(len(curr_tree['children'])):
                trees.append(curr_tree['children'][i])
    return full_img

def upsample_from_quadtree(quadtree,full_img,device="cuda"):
    # Guaranteed to be a leaf node
    data = quadtree['data']
    upscale_factor = int(quadtree['stride'])
    for x in range(quadtree['x_start'], quadtree['x_start']+data.shape[0]*upscale_factor, upscale_factor):
        for y in range(quadtree['y_start'], quadtree['y_start']+data.shape[1]*upscale_factor, upscale_factor):
            full_img[x:x+upscale_factor,y:y+upscale_factor,:] = \
            data[int((x-quadtree['x_start'])/upscale_factor), 
            int((y-quadtree['y_start'])/upscale_factor), :]
    return full_img

def pointwise_upscaling(img):
    upscaled_img = torch.zeros([int(img.shape[0]*2), int(img.shape[1]*2), 
    img.shape[2]]).to(img.device)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            upscaled_img[x*2:(x+1)*2, y*2:(y+1)*2,:] = img[x,y,:]
    return upscaled_img

def upscale_from_quadtree_debug(quadtree,max_stride=8,device="cuda"):
    cmap = [
        torch.tensor(np.array([0, 0, 0])),
        torch.tensor(np.array([37, 15, 77])),
        torch.tensor(np.array([115, 30, 107])),
        torch.tensor(np.array([178, 52, 85])),
        torch.tensor(np.array([233, 112, 37])),
        torch.tensor(np.array([244, 189, 55])),
        torch.tensor(np.array([247, 251, 162])),
        torch.tensor(np.array([255, 255, 255]))
    ]
    full_img = torch.zeros([int(quadtree['full_shape'][0]/max_stride),
    int(quadtree['full_shape'][1]/max_stride), quadtree['full_shape'][2]]).to(device)
    curr_stride = max_stride
    while(curr_stride > 0):
        x = int(np.log(curr_stride) / np.log(2))
        c = cmap[-1-x].to(device)
        # 1. Fill in known data
        trees = [quadtree]
        while(len(trees) > 0):
            curr_tree = trees.pop(0)
            if(curr_tree['data'] is not None and curr_tree['stride'] <= curr_stride):
                full_img[
                    int(curr_tree['x_start']/curr_stride): \
                    int(curr_tree['x_start']/curr_stride)+int((curr_tree['data'].shape[0]*curr_tree['stride'])/curr_stride),
                    int(curr_tree['y_start']/curr_stride): \
                    int(curr_tree['y_start']/curr_stride)+int((curr_tree['data'].shape[1]*curr_tree['stride'])/curr_stride),
                    :
                ] = c
            elif(len(curr_tree['children']) > 0):
                for i in range(len(curr_tree['children'])):
                    trees.append(curr_tree['children'][i])

        # 2. Upsample if necessary
        if(curr_stride > 1):
            full_img = pointwise_upscaling(full_img)
        
        curr_stride = int(curr_stride / 2)

    return full_img

def upscale_from_quadtree_with_seams(quadtree,device="cuda"):
    full_img = torch.zeros([int(quadtree['full_shape'][0]),
    int(quadtree['full_shape'][1]), quadtree['full_shape'][2]]).to(device)
    
    # 1. Fill in known data
    trees = [quadtree]
    while(len(trees) > 0):
        curr_tree = trees.pop(0)
        if(curr_tree['data'] is not None):

            img_part = F.interpolate(curr_tree['data'].permute(2, 0, 1).unsqueeze(0), 
            scale_factor=curr_tree['stride'], mode='bilinear', align_corners=True)[0].permute(1, 2, 0)

            full_img[
                int(curr_tree['x_start']):int(curr_tree['x_start'])+img_part.shape[0],
                int(curr_tree['y_start']):int(curr_tree['y_start'])+img_part.shape[1],
                :
            ] = img_part
        elif(len(curr_tree['children']) > 0):
            for i in range(len(curr_tree['children'])):
                trees.append(curr_tree['children'][i])

    #imageio.imwrite(str(curr_stride)+".jpg", full_img.cpu().numpy())
    return full_img

'''
def upscale_from_quadtree_start(quadtree,max_stride=8,device="cuda"):
    full_img = torch.zeros([int(quadtree['full_shape'][0]/max_stride),
    int(quadtree['full_shape'][1]/max_stride), quadtree['full_shape'][2]]).to(device)
    
    curr_stride = max_stride
    while(curr_stride > 0):
        # 1. Fill in known data
        trees = [quadtree]
        while(len(trees) > 0):
            curr_tree = trees.pop(0)
            if(curr_tree['data'] is not None and curr_tree['stride'] <= curr_stride):
                full_img[
                    int(curr_tree['x_start']/curr_stride): \
                    int(curr_tree['x_start']/curr_stride)+int((curr_tree['data'].shape[0]*curr_tree['stride'])/curr_stride),
                    int(curr_tree['y_start']/curr_stride): \
                    int(curr_tree['y_start']/curr_stride)+int((curr_tree['data'].shape[1]*curr_tree['stride'])/curr_stride),
                    :
                ] = AvgPool2D(curr_tree['data'].permute(2, 0, 1).unsqueeze(0), 
                int(curr_stride/curr_tree['stride']))[0].permute(1, 2, 0)
            elif(len(curr_tree['children']) > 0):
                for i in range(len(curr_tree['children'])):
                    trees.append(curr_tree['children'][i])

        #imageio.imwrite(str(curr_stride)+"pre_upsample.jpg", full_img.cpu().numpy())
        if(curr_stride > 1):
            # 2. Upsample
            full_img = full_img.permute(2,0,1).unsqueeze(0)
            full_img = F.interpolate(full_img, scale_factor=2, mode='bilinear', align_corners=True)
            full_img = full_img[0].permute(1,2,0)
            #imageio.imwrite(str(curr_stride)+"post_upsample.jpg", full_img.cpu().numpy())
        curr_stride = int(curr_stride / 2)
                
    #imageio.imwrite(str(curr_stride)+".jpg", full_img.cpu().numpy())
    return full_img
'''
def upscale_from_quadtree_start(quadtree,max_stride=8,device="cuda"):
    full_imgs, masks = quadtree_to_img_downscaled(quadtree, max_stride, device)
    
    curr_stride = max_stride
    full_img = full_imgs[0]
    i = 0
    while(curr_stride > 1):  
        # 1. Upsample
        full_img = full_img.permute(2,0,1).unsqueeze(0)
        full_img = F.interpolate(full_img, scale_factor=2, mode='bilinear', align_corners=True)
        full_img = full_img[0].permute(1,2,0)
        
        curr_stride = int(curr_stride / 2)
        i += 1

        # 2. Fill in data
        full_img[masks[i] > 0] = full_imgs[i][masks[i] > 0]


    #imageio.imwrite(str(curr_stride)+".jpg", full_img.cpu().numpy())
    return full_img

def quadtree_to_img_downscaled(quadtree, stride,device="cuda"):
    GT_data = []
    masks = []
    full_img = torch.zeros(quadtree['full_shape']).to(device)
    mask = torch.zeros(full_img.shape).to(device)
    curr_stride = 1
    
    while(curr_stride <= stride):
        # 1. Fill in known data
        trees = [quadtree]
        while(len(trees) > 0):
            curr_tree = trees.pop(0)
            if(curr_tree['data'] is not None and curr_tree['stride'] == curr_stride):
                full_img[
                    int(curr_tree['x_start']/curr_stride): \
                    int(curr_tree['x_start']/curr_stride)+int((curr_tree['data'].shape[0]*curr_tree['stride'])/curr_stride),
                    int(curr_tree['y_start']/curr_stride): \
                    int(curr_tree['y_start']/curr_stride)+int((curr_tree['data'].shape[1]*curr_tree['stride'])/curr_stride),
                    :
                ] = curr_tree['data']
                mask[
                    int(curr_tree['x_start']/curr_stride): \
                    int(curr_tree['x_start']/curr_stride)+int((curr_tree['data'].shape[0]*curr_tree['stride'])/curr_stride),
                    int(curr_tree['y_start']/curr_stride): \
                    int(curr_tree['y_start']/curr_stride)+int((curr_tree['data'].shape[1]*curr_tree['stride'])/curr_stride),
                    :
                ] = torch.ones(curr_tree['data'].shape)
            elif(len(curr_tree['children']) > 0):
                for i in range(len(curr_tree['children'])):
                    trees.append(curr_tree['children'][i])
        GT_data.insert(0, full_img.clone())
        masks.insert(0, mask.clone())

        if(curr_stride < stride):
            # 2. Downsample
            full_img = full_img.permute(2,0,1).unsqueeze(0)
            full_img = AvgPool2D(full_img, 2)
            full_img = full_img[0].permute(1,2,0)
            mask = mask[::2, ::2, :]

        curr_stride = int(curr_stride * 2)

        
                
    #imageio.imwrite(str(curr_stride)+".jpg", full_img.cpu().numpy())
    return GT_data, masks

def conditional_downsample_quadtree(img,GT_image,criterion,criterion_value,
min_chunk_size=32,max_stride=8,device="cuda"):
    quadtrees_to_check = [img]
    trees_checked = 0
    #sequence_of_downsampling = []
    #sequence_of_downsampling_debug = []
    #sequence_of_downsampling_debug.append(upscale_from_quadtree_debug(img,max_stride=max_stride,device=device).cpu().numpy())
    #sequence_of_downsampling.append(upsample_from_quadtree_start(img,device=device).cpu().numpy())
    while(len(quadtrees_to_check) > 0):
        trees_checked += 1
        t = quadtrees_to_check.pop(0)

        # Check if we can downsample this leaf node
        t['stride'] = int(t['stride']*2)
        original_data = t['data'].clone()
        downsampled_data = original_data[::2,::2,:].clone()
        t['data'] = downsampled_data
        new_img = upscale_from_quadtree_start(img,max_stride=max_stride,device=device)
        
        # If criterion not met, reset data and stride, and see
        # if the node is large enough to split into subnodes
        # Otherwise, we keep the downsample, and add the node back as a 
        # leaf node
        if(not criterion(GT_image, new_img, criterion_value)):
            t['data'] = original_data
            t['stride'] = int(t['stride'] / 2)
            if(t['data'].shape[0] > min_chunk_size):
                for x_quad_start in range(0, t['data'].shape[0], int(t['data'].shape[0]/2)):
                    for y_quad_start in range(0, int(t['data'].shape[1]), int(t['data'].shape[1]/2)):
                        t_new = {
                            "full_shape": [int(t['full_shape'][0]/2), int(t['full_shape'][1]/2), t['full_shape'][2]],
                            "data": t['data'][x_quad_start:x_quad_start+int(t['data'].shape[0]/2),
                            y_quad_start:y_quad_start+int(t['data'].shape[1]/2),:].clone(),
                            "stride": int(t['stride']),
                            "x_start": int(x_quad_start*t['stride'] + t['x_start']),
                            "y_start": int(y_quad_start*t['stride'] + t['y_start']),
                            "children": []
                        }
                        t['children'].append(t_new)     
                        quadtrees_to_check.append(t_new)                   
                t['data'] = None
                t['stride'] = None
                t['x_start'] = None
                t['y_start'] = None
        else:
            #sequence_of_downsampling_debug.append(upscale_from_quadtree_debug(img,max_stride=max_stride,device=device).cpu().numpy())
            #sequence_of_downsampling.append(upsample_from_quadtree_start(img,device=device).cpu().numpy())
            if(t['stride'] < max_stride):
                quadtrees_to_check.append(t)
    #imageio.mimwrite("downsampling_sequence.gif", sequence_of_downsampling, fps=5)
    #imageio.mimwrite("downsampling_sequence_debug.gif", sequence_of_downsampling_debug, fps=5)
    print("Trees checked: %i" % trees_checked)
    return img            

def conditional_pooling_quadtree(img, GT_image, criterion, criterion_value,
min_chunk_size : int, max_stride : int, device : str):
    quadtrees_to_check = [img]
    trees_checked = 0
    #sequence_of_downsampling = []
    #sequence_of_downsampling_debug = []
    #sequence_of_downsampling_debug.append(upscale_from_quadtree_debug(img,max_stride=max_stride,device=device).cpu().numpy())
    #sequence_of_downsampling.append(upsample_from_quadtree_start(img,device=device).cpu().numpy())
    while(len(quadtrees_to_check) > 0):
        trees_checked += 1
        t = quadtrees_to_check.pop(0)

        # Check if we can downsample this leaf node
        t['stride'] = int(t['stride']*2)
        original_data = t['data'].clone()
        downsampled_data = AvgPool2D(original_data.clone().permute(2, 0, 1).unsqueeze(0),
        2)[0].permute(1, 2, 0)
        t['data'] = downsampled_data
        new_img = upscale_from_quadtree_start(img,device=device)
        
        # If criterion not met, reset data and stride, and see
        # if the node is large enough to split into subnodes
        # Otherwise, we keep the downsample, and add the node back as a 
        # leaf node
        if(not criterion(GT_image, new_img, criterion_value)):
            t['data'] = original_data
            t['stride'] = int(t['stride'] / 2)
            if(t['data'].shape[0] > min_chunk_size):
                for x_quad_start in range(0, t['data'].shape[0], int(t['data'].shape[0]/2)):
                    for y_quad_start in range(0, int(t['data'].shape[1]), int(t['data'].shape[1]/2)):
                        t_new = {
                            "full_shape": [int(t['full_shape'][0]/2), int(t['full_shape'][1]/2), t['full_shape'][2]],
                            "data": t['data'][x_quad_start:x_quad_start+int(t['data'].shape[0]/2),
                            y_quad_start:y_quad_start+int(t['data'].shape[1]/2),:].clone(),
                            "stride": int(t['stride']),
                            "x_start": int(x_quad_start*t['stride'] + t['x_start']),
                            "y_start": int(y_quad_start*t['stride'] + t['y_start']),
                            "children": []
                        }
                        t['children'].append(t_new)     
                        quadtrees_to_check.append(t_new)                   
                t['data'] = None
                t['stride'] = None
                t['x_start'] = None
                t['y_start'] = None
        else:
            #sequence_of_downsampling_debug.append(upscale_from_quadtree_debug(img,max_stride=max_stride,device=device).cpu().numpy())
            #sequence_of_downsampling.append(upsample_from_quadtree_start(img,device=device).cpu().numpy())
            if(t['stride'] < max_stride and t['data'].shape[0] > 1):
                quadtrees_to_check.append(t)
    #imageio.mimwrite("downsampling_sequence.gif", sequence_of_downsampling, fps=5)
    #imageio.mimwrite("downsampling_sequence_debug.gif", sequence_of_downsampling_debug, fps=5)
    print("Trees checked: %i" % trees_checked)
    return img            


def ssim_criterion(GT_image, img, min_ssim=0.6):
    return ssim(img.permute(2, 0, 1).unsqueeze(0), GT_image.permute(2, 0, 1).unsqueeze(0)) > min_ssim

@torch.jit.script
def psnr_criterion(GT_image, img, min_PSNR : float):
    return PSNR(img, GT_image) > min_PSNR

@torch.jit.script
def mse_criterion(GT_image, img, max_mse : float):
    return MSE(img, GT_image) < max_mse



max_stride = 16
min_chunk = 8
criterion = psnr_criterion
criterion_value = 90
device="cuda"

img_gt = torch.from_numpy(imageio.imread("TestingData/quadtree_images/mixing.jpg").astype(np.float32)).to(device)
#img_gt = AvgPool2D(img_gt.permute(2, 0, 1).unsqueeze(0), 16)[0].permute(1, 2, 0)
img = {
    "full_shape": img_gt.shape, 
    "data": img_gt,
    "stride": 1,
    "x_start": 0,
    "y_start": 0,
    "children": []
    }
torch.save(img, './Output/img_full.torch')


##############################################
img = torch.load("./Output/img_full.torch")
start_time = time.time()
img_quadtree = conditional_downsample_quadtree(img,img_gt,criterion,criterion_value,
min_chunk_size=min_chunk,max_stride=max_stride,device="cuda")
end_time = time.time()
print("Conditional downsample took % 0.02f seconds" % (end_time - start_time))
torch.save(img_quadtree, "./Output/img_downsample_quadtree.torch")

img_upscaled = upscale_from_quadtree_start(img_quadtree,max_stride=max_stride,device=device)
img_upsampled = upsample_from_quadtree_start(img_quadtree,device=device)
img_upscaled_seams = upscale_from_quadtree_with_seams(img_quadtree,device=device)
debug = upscale_from_quadtree_debug(img_quadtree,max_stride=max_stride,device=device)

imageio.imwrite("./Output/img_downsample_upscaled.jpg", img_upscaled.cpu().numpy())
imageio.imwrite("./Output/img_downsample_upsampled.jpg", img_upsampled.cpu().numpy())
imageio.imwrite("./Output/img_downsample_upsampled_debug.jpg", debug.cpu().numpy())
imageio.imwrite("./Output/img_downsample_upsampled_seams.jpg", img_upscaled_seams.cpu().numpy())

################################################
img = torch.load("./Output/img_full.torch")
start_time = time.time()
img_quadtree = conditional_pooling_quadtree(img,img_gt,criterion,criterion_value,
min_chunk_size=min_chunk,max_stride=max_stride,device="cuda")
end_time = time.time()
print("Conditional pooling took % 0.02f seconds" % (end_time - start_time))
torch.save(img_quadtree, "./Output/img_pooling_quadtree.torch")

img_upscaled = upscale_from_quadtree_start(img_quadtree,max_stride=max_stride,device=device)
img_upsampled = upsample_from_quadtree_start(img_quadtree,device=device)
img_upscaled_seams = upscale_from_quadtree_with_seams(img_quadtree,device=device)
debug = upscale_from_quadtree_debug(img_quadtree,max_stride=max_stride,device=device)

imageio.imwrite("./Output/img_pooling_upscaled.jpg", img_upscaled.cpu().numpy())
imageio.imwrite("./Output/img_pooling_upsampled.jpg", img_upsampled.cpu().numpy())
imageio.imwrite("./Output/img_pooling_upsampled_debug.jpg", debug.cpu().numpy())
imageio.imwrite("./Output/img_pooling_upsampled_seams.jpg", img_upscaled_seams.cpu().numpy())




