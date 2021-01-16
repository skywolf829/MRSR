import os
import imageio
import argparse
from typing import Union, Tuple
import numpy as np
import zeep
import struct
import base64
import time
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token

result=client.service.GetAnyCutoutWeb(token,"isotropic1024coarse", "u", 100,
                                            1, 1, 
                                            1, 100, 100, 100,
                                            1, 1, 1, 0, "")  # put empty string for the last parameter
# transfer base64 format to numpy
nx=int(100)
ny=int(100)
nz=int(100)
base64_len=int(nx*ny*nz*3)
base64_format='<'+str(base64_len)+'f'

result=struct.unpack(base64_format, result)
result=np.array(result).reshape((nz, ny, nx, 3))
print(result.shape)