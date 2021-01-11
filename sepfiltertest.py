import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)

def separable_3d():
    return nn.Sequential(
        nn.Conv3d(64, 64, (3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
        nn.Conv3d(64, 64, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
        #nn.Conv3d(64, 64, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
    )

num_layers = 15
iters = 1000

modules = []
for i in range(num_layers):
    modules.append(
        nn.Conv3d(64, 64, (3, 3, 3), 
        stride=(1, 1, 1), padding=(1, 1, 1))
    )
model1 = nn.Sequential(*modules).cuda()

modules = []
for i in range(num_layers):
    modules.append(
        separable_3d()
    )
model2 = nn.Sequential(*modules).cuda()


model1.apply(weights_init)
model2.apply(weights_init)

t = torch.rand([1, 64, 25, 25, 25]).cuda()
model1(t)
model2(t)

start_t = time.time()
for i in range(iters):
    model1(t)
print("Took %0.05f sec for %i iterations on model with %i full 3D conv layers" % \
(time.time() - start_t, iters, num_layers))

t = torch.rand([1, 64, 25, 25, 25]).cuda()
start_t = time.time()
for j in range(iters):
    model2(t)
print("Took %0.05f sec for %i iterations on model with %i seperable 3D conv layers" % \
(time.time() - start_t, iters, num_layers))