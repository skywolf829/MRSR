import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return JSONEncoder.default(self, obj)

def MSE(x, GT):
    return ((x-GT)**2).mean()

def PSNR(x, GT, max_diff=255.0):
    return 20 * torch.log(torch.tensor(max_diff)) - 10*torch.log(MSE(x, GT))

#old
def up(img, x_start, x_end, y_start, y_end, skip):
    for x in range(x_start, x_end, skip):
        for y in range(y_start, y_end, skip):
            img[x:x+skip,y:y+skip,:] = img[x,y,:]
    return img

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


def upscale_from_quadtree_debug(quadtree,max_stride=8,device="cuda"):
    full_img = torch.zeros([int(quadtree['full_shape'][0]/max_stride),
    int(quadtree['full_shape'][1]/max_stride), quadtree['full_shape'][2]]).to(device)
    
    curr_stride = max_stride
    while(curr_stride > 1):
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
                ] = torch.tensor([curr_stride, curr_stride, curr_stride]).to(device)
            elif(len(curr_tree['children']) > 0):
                for i in range(len(curr_tree['children'])):
                    trees.append(curr_tree['children'][i])

        # 2. Upsample
        full_img = full_img.permute(2,0,1).unsqueeze(0)
        full_img = F.interpolate(full_img, scale_factor=2, mode='bilinear', align_corners=True)
        full_img = full_img[0].permute(1,2,0)
        curr_stride = int(curr_stride / 2)

    # Fill in known data again
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
            ] =  torch.tensor([curr_stride, curr_stride, curr_stride]).to(device)
        elif(len(curr_tree['children']) > 0):
            for i in range(len(curr_tree['children'])):
                trees.append(curr_tree['children'][i])
    return full_img

def upscale_from_quadtree_start(quadtree,max_stride=8,device="cuda"):
    full_img = torch.zeros([int(quadtree['full_shape'][0]/max_stride),
    int(quadtree['full_shape'][1]/max_stride), quadtree['full_shape'][2]]).to(device)
    
    curr_stride = max_stride
    while(curr_stride > 1):
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
                ] = curr_tree['data'][::int(curr_stride/curr_tree['stride']),::int(curr_stride/curr_tree['stride']),:]
            elif(len(curr_tree['children']) > 0):
                for i in range(len(curr_tree['children'])):
                    trees.append(curr_tree['children'][i])

        # 2. Upsample
        full_img = full_img.permute(2,0,1).unsqueeze(0)
        full_img = F.interpolate(full_img, scale_factor=2, mode='bilinear', align_corners=True)
        full_img = full_img[0].permute(1,2,0)
        curr_stride = int(curr_stride / 2)

    # Fill in known data again
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
            ] = curr_tree['data'][::int(curr_stride/curr_tree['stride']),::int(curr_stride/curr_tree['stride']),:]
        elif(len(curr_tree['children']) > 0):
            for i in range(len(curr_tree['children'])):
                trees.append(curr_tree['children'][i])
    return full_img

def upscale_from_quadtree(quadtree,full_img,device="cuda"):
    # Guaranteed to be a leaf node
    data = quadtree['data']
    upscale_factor = quadtree['stride']
    data = data.permute(2,0,1).unsqueeze(0)
    data = F.interpolate(data, scale_factor=upscale_factor, mode='bilinear', align_corners=True)
    data = data[0].permute(1,2,0)
    full_img[quadtree['x_start']:quadtree['x_start']+data.shape[0], 
        quadtree['y_start']:quadtree['y_start']+data.shape[1],:] = data
    return full_img

def conditional_downsample_quadtree(img,GT_image,max_MSE=3.0,min_ssim=0.9,min_PSNR=30,
max_chunk_size=32,max_stride=8,device="cuda"):
    # Assume starting with a quadtree(img) that is a leaf node
    quadtrees_to_check = [img]
    trees_checked = 0
    while(len(quadtrees_to_check) > 0):
        trees_checked += 1
        t = quadtrees_to_check.pop(0)
        if(t['stride'] < max_stride and (t['data'] is None or t['data'].shape[0] > max_chunk_size)):
            #print("Checking quadtree in quadrant %i:%i, %i:%i" % (t['x_start'], 
            #t['x_start']+t['data'].shape[0]*t['stride'], t['y_start'], t['y_start']+t['data'].shape[1]*t['stride']))

            if(len(t['children']) == 0):
                #print("Splitting into 4 subtrees to check")
                # if the tree has no children, split it into 4 so they can be checked
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
                        
                t['data'] = None
                quadtrees_to_check.append(t)
            else:
                # if the tree does have children, see if the stride can be increased in any quadtree without going above 
                # the max mse / min ssim
                
                for i in range(len(t['children'])):
                    #print("Attempting to downscale tree in quadrant %i:%i, %i:%i" % \
                    #(t['children'][i]['x_start'], 
                    #t['children'][i]['x_start']+t['children'][i]['data'].shape[0]*t['children'][i]['stride'], 
                    #t['children'][i]['y_start'], 
                    #t['children'][i]['y_start']+t['children'][i]['data'].shape[1]*t['children'][i]['stride']))

                    t['children'][i]['stride'] = int( t['children'][i]['stride']*2)
                    child_original_data = t['children'][i]['data']
                    child_downsampled_data = child_original_data[::2,::2,:]
                    t['children'][i]['data'] = child_downsampled_data
                    new_img = upscale_from_quadtree_start(img,device=device)

                    #print("This downscaled subtree gives MSE %0.03f, SSIM %0.02f" % (m,s))
                    '''
                    if(s > min_ssim):
                        #print("This SSIM is acceptable, added to list to check later")
                        quadtrees_to_check.append(t['children'][i])
                    else:
                        #print("SSIM too low, reverting dowscaling")
                        t['children'][i]['data'] = child_original_data
                        t['children'][i]['stride'] = int(t['children'][i]['stride'] / 2)
                        quadtrees_to_check.append(t['children'][i])
                    '''
                    '''
                    m = MSE(new_img, GT_image)
                    if(m > max_MSE):
                        #print("MSE too high, reverting dowscaling")
                        t['children'][i]['data'] = child_original_data
                        t['children'][i]['stride'] =  int(t['children'][i]['stride'] / 2)
                    quadtrees_to_check.append(t['children'][i])
                    '''
                    p = PSNR(new_img, GT_image)
                    if(p < min_PSNR):
                        #print("PSNR too low, reverting dowscaling")
                        t['children'][i]['data'] = child_original_data
                        t['children'][i]['stride'] =  int(t['children'][i]['stride'] / 2)

                    quadtrees_to_check.append(t['children'][i])
                    
    print("Trees checked: %i" % trees_checked)
    return img            

device="cuda"
snick_gt = torch.from_numpy(imageio.imread("snickers.jpg").astype(np.float32)).to(device)
snick = {
    "full_shape": snick_gt.shape, 
    "data": snick_gt,
    "stride": 1,
    "x_start": 0,
    "y_start": 0,
    "children": []
    }
j = json.dumps(snick, cls=NumpyArrayEncoder)
f = open("snick.json",'w')
f.write(j)
f.close()
import time
start_time = time.time()
snick = conditional_downsample_quadtree(snick,snick_gt,max_MSE=3.0,min_ssim=0.9,min_PSNR=60,
max_chunk_size=32,max_stride=8,device="cuda")
end_time = time.time()

print("Conditional downsample took % 0.02f seconds" % (end_time - start_time))
j = json.dumps(snick, cls=NumpyArrayEncoder)
f = open("snick_quadtree.json",'w')
f.write(j)
f.close()


snick_upscaled = upscale_from_quadtree_start(snick)

print("Final PSNR: %0.02f" % PSNR(snick_gt, snick_upscaled))

snick_upsampled = upsample_from_quadtree_start(snick)
debug = upscale_from_quadtree_debug(snick)
imageio.imwrite("upscaled_snick.jpg", snick_upscaled.cpu().numpy())
imageio.imwrite("upsampled_snick.jpg", snick_upsampled.cpu().numpy())
imageio.imwrite("upsampled_snick_debug.jpg", debug.cpu().numpy())