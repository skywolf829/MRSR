import torch
from utility_functions import AvgPool3D

a = torch.ones([1, 3, 128, 128, 128])

print(a.shape)
b = AvgPool3D(a)

print(b.shape)