import torch
from utility_functions import AvgPool3D
import h5py
import imageio
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('bigboy.h5', 'r')
data = torch.tensor(f['data']).type(torch.FloatTensor).cuda()
f.close()
data = data.unsqueeze(0)
data_mag = torch.linalg.norm(data,axis=1)[0]

data_mag /= data_mag.max()
image_out = torch.zeros(data_mag.shape).cuda()
image_out = image_out.unsqueeze(2)
image_out = image_out.repeat(1, 1, 3)

#plt.hist(data_mag.cpu().numpy().flatten(), bins=25, density=True, cumulative=True)
#plt.show()

'''
# black white red
color_mapping_keys = [0.01, 0.3, 0.6]
color_mapping_values = [torch.from_numpy(np.array([0.0, 0.0, 0.0])).type(torch.FloatTensor).cuda(), 
                        torch.from_numpy(np.array([200.0, 200.0, 200.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([200.0, 0.0, 0.0])).type(torch.FloatTensor).cuda()]
'''
'''
# rainbow               R   O    Y     G    B    I     V     R
color_mapping_keys = [0.0, 0.08, 0.16, 0.25, 0.3, 0.35, 0.4, 1.0]
color_mapping_values = [torch.from_numpy(np.array([128, 0.0, 0.0])).type(torch.FloatTensor).cuda(), 
                        torch.from_numpy(np.array([255.0, 136.0, 0.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([255.0, 255.0, 0.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([0.0, 255.0, 0.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([0.0, 0.0, 255.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([0.0, 255.0, 200.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([128.0, 76.0, 128.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([128.0, 0.0, 0.0])).type(torch.FloatTensor).cuda()]
'''
# another             blk       Y     G    B    I     V     R
color_mapping_keys = [0.0, 0.16, 0.4, 0.6]
color_mapping_values = [torch.from_numpy(np.array([9.0, 171.0, 166.0])).type(torch.FloatTensor).cuda(), 
                        torch.from_numpy(np.array([0.0, 0.0, 0.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([121.0, 9.0, 9.0])).type(torch.FloatTensor).cuda(),
                        torch.from_numpy(np.array([255.0, 255.0, 255.0])).type(torch.FloatTensor).cuda(),
                        ]
image_out[data_mag < color_mapping_keys[0]] = color_mapping_values[0]

for i in range(len(color_mapping_keys)-1):
    ratios = data_mag.clone()
    ratios -= color_mapping_keys[i]
    ratios *= 1 / (color_mapping_keys[i+1] - color_mapping_keys[i])
    ratios = ratios.type(torch.FloatTensor).cuda()
    
    cmap = (1-ratios).view(ratios.shape[0], ratios.shape[0], 1).repeat(1, 1, 3) \
         * color_mapping_values[i].view(1, 1, 3).repeat(ratios.shape[0], ratios.shape[1], 1) 
    cmap += color_mapping_values[i+1].view(1, 1, 3).repeat(ratios.shape[0], ratios.shape[1], 1) \
         * ratios.view(ratios.shape[0], ratios.shape[0], 1).repeat(1, 1, 3)
    
    indices = torch.bitwise_and(data_mag >= color_mapping_keys[i],
    data_mag < color_mapping_keys[i+1])
    image_out[indices] = cmap[indices]

image_out[data_mag > color_mapping_keys[-1]] = color_mapping_values[-1]

imageio.imwrite("bigboy.jpg", image_out.cpu().numpy())