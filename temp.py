import torch
import torch.nn.functional as F
from utility_functions import AvgPool3D
import h5py
import imageio
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import AvgPool2D




# Experiment to see if the distribution of downscaled frames that are
# downscaled with a method that doesn't follow downscale(x, S) = 
# downscale(downscale(x, S/2), S/2).
# Compare distributions with PCA? T-SNE? Just mean and variance?
#
from download_JHUTDB import get_full_frame_parallel


frames = []
name = "isotropic1024"
startts = 1
endts = 1001
ts_skip = 10
ds = 32

ds_once_data = []
ds_many_data = []

ds_once_stats = []
ds_many_stats = []

for i in range(startts, endts, ts_skip):
     print("TS " + str(i))
     f = get_full_frame_parallel(0, 1024, 1,#x
     0, 1024, 1, #y
     512, 513, 1, #z
     name, i, 
     "u", 3, 
     64)
     f = f[:,:,0,:].astype(np.float32)

     f_img = f.copy()
     f_img[:,:,0] -= f_img[:,:,0].min()
     f_img[:,:,0] *= (255.0/f_img[:,:,0].max())
     f_img[:,:,1] -= f_img[:,:,1].min()
     f_img[:,:,1] *= (255.0/f_img[:,:,1].max())
     f_img[:,:,2] -= f_img[:,:,2].min()
     f_img[:,:,2] *= (255.0/f_img[:,:,2].max())
     f_img = f_img.astype(np.uint8)
     imageio.imwrite("full_res.png", f_img)

     f = f.swapaxes(0,2).swapaxes(1,2)
     f = torch.from_numpy(f).unsqueeze(0)
     f_downscaled_once = F.interpolate(f.clone(), mode="bilinear", align_corners=True, scale_factor=1/ds)
     f_downscaled_many = f.clone()
     curr_s = 1
     while(curr_s < ds):
          f_downscaled_many = F.interpolate(f_downscaled_many, mode="bilinear", align_corners=True, scale_factor=1/2)
          curr_s *= 2
     
     ds_once_data.append(f_downscaled_once.clone().view(1, -1).cpu().numpy())
     ds_many_data.append(f_downscaled_many.clone().view(1, -1).cpu().numpy())
     ds_once_stats.append(np.array([f_downscaled_once.min(), f_downscaled_once.max(), f_downscaled_once.mean(), f_downscaled_once.std()]))
     ds_many_stats.append(np.array([f_downscaled_many.min(), f_downscaled_many.max(), f_downscaled_many.mean(), f_downscaled_many.std()]))

     print("DS_once min/max: %0.03f/%0.03f, mean/std: %0.03f/%0.03f" % \
          (f_downscaled_once.min(), f_downscaled_once.max(), f_downscaled_once.mean(), f_downscaled_once.std()))
     print("DS_many min/max: %0.03f/%0.03f, mean/std: %0.03f/%0.03f" % \
          (f_downscaled_many.min(), f_downscaled_many.max(), f_downscaled_many.mean(), f_downscaled_many.std()))

     ds_once_img = f_downscaled_once[0].permute(1, 2, 0).cpu().numpy()
     ds_once_img[:,:,0] -= ds_once_img[:,:,0].min()
     ds_once_img[:,:,0] *= (255.0/ds_once_img[:,:,0].max())
     ds_once_img[:,:,1] -= ds_once_img[:,:,1].min()
     ds_once_img[:,:,1] *= (255.0/ds_once_img[:,:,1].max())
     ds_once_img[:,:,2] -= ds_once_img[:,:,2].min()
     ds_once_img[:,:,2] *= (255.0/ds_once_img[:,:,2].max())
     ds_once_img = ds_once_img.astype(np.uint8)
     imageio.imwrite("downscaled_once.png", ds_once_img)

     ds_many_img = f_downscaled_many[0].permute(1, 2, 0).cpu().numpy()
     ds_many_img[:,:,0] -= ds_many_img[:,:,0].min()
     ds_many_img[:,:,0] *= (255.0/ds_many_img[:,:,0].max())
     ds_many_img[:,:,1] -= ds_many_img[:,:,1].min()
     ds_many_img[:,:,1] *= (255.0/ds_many_img[:,:,1].max())
     ds_many_img[:,:,2] -= ds_many_img[:,:,2].min()
     ds_many_img[:,:,2] *= (255.0/ds_many_img[:,:,2].max())
     ds_many_img = ds_many_img.astype(np.uint8)
     imageio.imwrite("downscaled_many.png", ds_many_img)

ds_once_stats = np.array(ds_once_stats)
ds_many_stats = np.array(ds_many_stats)  


ds_once_data = np.concatenate(ds_once_data, axis=0)
ds_many_data = np.concatenate(ds_many_data, axis=0)
all_data = np.concatenate([ds_once_data, ds_many_data], axis=0)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(all_data)
all_data_transformed = pca.transform(all_data)

plt.scatter(all_data_transformed[:ds_once_data.shape[0],0], all_data_transformed[:ds_once_data.shape[0],1], 
color='red', label='downscaled once', marker='x',alpha=0.5)

plt.scatter(all_data_transformed[ds_once_data.shape[0]:,0], all_data_transformed[ds_once_data.shape[0]:,1], 
color='blue', label='downscaled many times', marker='o',alpha=0.5)
plt.legend()
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.title("PCA decomposition of 2D "+str(ds)+"x downscaled fluid frames slices")
plt.show()

plt.clf()

plt.plot()
plt.plot(np.arange(0, ds_once_stats.shape[0]), ds_once_stats[:,0], marker='x', label='downscaled once minimum velocity component')
plt.plot(np.arange(0, ds_many_stats.shape[0]), ds_many_stats[:,0], marker='o', label='downscaled many minimum velocity component')
plt.plot(np.arange(0, ds_once_stats.shape[0]), ds_once_stats[:,1], marker='x', label='downscaled once maximum velocity component')
plt.plot(np.arange(0, ds_many_stats.shape[0]), ds_many_stats[:,1], marker='o', label='downscaled many maximum velocity component')
plt.plot(np.arange(0, ds_once_stats.shape[0]), ds_once_stats[:,2], marker='x', label='downscaled once mean velocity component')
plt.plot(np.arange(0, ds_many_stats.shape[0]), ds_many_stats[:,2], marker='o', label='downscaled many mean velocity component')
plt.plot(np.arange(0, ds_once_stats.shape[0]), ds_once_stats[:,3], marker='x', label='downscaled once std of velocity component')
plt.plot(np.arange(0, ds_many_stats.shape[0]), ds_many_stats[:,3], marker='o', label='downscaled many std of velocity component')
plt.legend()
plt.xlabel("Simulation timestep")
plt.ylabel("m/s")
plt.title("Min/max/mean/std of data downscaled by a factor of " + str(ds) + "x once or a factor of 2x " + str(int(np.log(ds)/np.log(2))) + " times")
plt.show()

'''
# This experiment shows the seams between leaf nodes of a quadtree
# when they are upscaled separately

skip = 32
ds = 8
a = imageio.imread("./TestingData/quadtree_images/Lenna.jpg").astype(np.float32)
b = torch.tensor(a).cuda().permute(2, 0, 1).unsqueeze(0)
c = F.interpolate(b.clone()[:,:,::ds,::ds], mode="bilinear", scale_factor=ds, align_corners=True)
c = c[0].permute(1, 2, 0).cpu().numpy()
imageio.imwrite("Lenna_noseams.jpg", c)

a[::skip, :, :] = np.array([0, 0, 0])
a[:, ::skip, :] = np.array([0, 0, 0])


imageio.imwrite("Lenna_cutput.jpg", a)

for x in range(0, b.shape[2], skip):
     for y in range(0, b.shape[3], skip):
          b[:,:,x:x+skip,y:y+skip] = F.interpolate(b[:,:,x:x+skip:ds,y:y+skip:ds], 
          scale_factor=ds, mode="bilinear", align_corners=True)
b = b[0].permute(1, 2, 0).cpu().numpy()
imageio.imwrite("Lenna_seams.jpg", b)

'''








'''
# This experiment tests which downscaling methods follow have the property
# downscale(x, S) = downscale(downscale(x, S/2), S/2)


a = torch.randn([1, 1, 16, 16]).cuda()
b = a.clone()

a = F.interpolate(a, scale_factor=0.5, mode='bilinear', align_corners=True)
a = F.interpolate(a, scale_factor=0.5, mode='bilinear', align_corners=True)
b = F.interpolate(b, scale_factor=0.25, mode='bilinear', align_corners=True)

print("Bilinear interpolation difference: " +str((b-a).sum()))

a = torch.randn([1, 1, 16, 16]).cuda()
b = a.clone()

a = F.interpolate(a, scale_factor=0.5, mode='bicubic', align_corners=True)
a = F.interpolate(a, scale_factor=0.5, mode='bicubic', align_corners=True)
b = F.interpolate(b, scale_factor=0.25, mode='bicubic', align_corners=True)

print("Bicubic interpolation difference: " +str((b-a).sum()))

a = torch.randn([1, 1, 16, 16]).cuda()
b = a.clone()

a = AvgPool2D(a, 2)
a = AvgPool2D(a, 2)
b = AvgPool2D(b, 4)

print("Avgerage pooling difference: " +str((b-a).sum()))

a = torch.randn([1, 1, 16, 16]).cuda()
b = a.clone()

a = a[:,:,::2,::2]
a = a[:,:,::2,::2]
b = b[:,:,::4,::4]

print("Subsampling difference: " +str((b-a).sum()))
'''

'''
f = h5py.File('bigboy.h5', 'r')
data = torch.tensor(f['data']).type(torch.FloatTensor).cuda()
f.close()
data = data.unsqueeze(0)
data_mag = torch.linalg.norm(data,axis=1)[0]

data_mag /= data_mag.max()
image_out = torch.zeros(data_mag.shape).cuda()
image_out = image_out.unsqueeze(2)
image_out = image_out.repeat(1, 1, 3)
'''
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

'''
# another             blk       Y     G    B    I     V     R
color_mapping_keys = [0.0, 0.21, 0.41, 0.6]
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

img = image_out.cpu().numpy().astype(np.uint8)
imageio.imwrite("bigboy.jpg", img)
'''