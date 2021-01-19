import imageio
import numpy as np
import torch
import torch.functional as F


def downsample_area(img, x_start, x_end, y_start, y_end, skip):
    
    for x in range(x_start, x_end, skip):
        for y in range(y_start, y_end, skip):
            img[x:x+skip,y:y+skip,:] = img[x,y,:]
    return img

def quadrant_downsample_PSNR_check(img, )

snick = imageio.imread("snickers.jpg")


snick_a = downsample_area(snick, 0, 512, 0, 512, 8)
snick_b = downsample_area(snick_a, 512, 1024, 0, 1024, 4)
imageio.imwrite("snickers_a.jpg", snick_a)
imageio.imwrite("snickers_b.jpg", snick_b)