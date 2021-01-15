from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a = torch.tensor([[1, 4, 2, 3], [1, 4, 2, 3],[1, 4, 2, 3],[0, 0, 0, 0]], dtype=float)
a = a.unsqueeze(0).unsqueeze(0)
print(a)
print(a.shape)
a_up = F.interpolate(a, mode='bilinear', scale_factor=2, align_corners=True)
print(a_up)

a_down = F.interpolate(a_up, mode='bilinear', scale_factor=0.5, align_corners=True)
print(a_down)