import torch
from utility_functions import AvgPool3D
import h5py
import imageio
import numpy as np

f = h5py.File('bigboy.h5', 'r')
data = torch.tensor(f['data']).cuda()
f.close()
data = data.unsqueeze(0)
print(data.shape)
data_mag = torch.linalg.norm(data,axis=1)[0]
print(data_mag.shape)

data_mag /= data_mag.max()
img_out = torch.zeros(data_mag.shape).cuda()
image_out = image_out.unsqueeze(2)
image_out = image_out.repeat(1, 1, 3)
print(image_out.shape)

color_mapping_keys = [0.1, 0.5, 0.9]
color_mapping_values = [torch.from_numpy(np.array([0, 0, 0])), 
                        torch.from_numpy(np.array([255, 255, 255])),
                        torch.from_numpy(np.array([255, 0, 0]))]

print((data_mag < 0.1).shape)
image_out[data_mag < color_mapping_keys[0]] = color_mapping_values[0]

for i in range(len(color_mapping_keys)-1):
    ratios = data_mag.clone()
    ratios -= color_mapping_keys[i]
    ratios *= 1 / (color_mapping_keys[i+1] - color_mapping_keys[i])
    image_out[data_mag >= color_mapping_keys[i] and data_mag < color_mapping_keys[i+1]] = \
    color_mapping_values[i] * (1-ratios) + color_mapping_values[i+1] * ratios

image_out[data_mag > color_mapping_keys[-1]] = color_mapping_values[-1]

imageio.imwrite("bigboy.png", image_out.cpu().numpy())