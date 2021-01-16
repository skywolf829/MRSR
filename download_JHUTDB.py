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
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')
token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token

def get_frame(x_start, x_end, x_step, 
y_start, y_end, y_step, 
z_start, z_end, z_step, 
sim_name, timestep, field, num_components):
    #print(x_start)
    #print(x_end)
    #print(y_start)
    #print(y_end)
    #print(z_start)
    #print(z_end)
    result=client.service.GetAnyCutoutWeb(token,sim_name, field, timestep,
                                            x_start+1, y_start+1, 
                                            z_start+1, x_end, y_end, z_end,
                                            x_step, y_step, z_step, 0, "")  # put empty string for the last parameter
    # transfer base64 format to numpy
    nx=int((x_end-x_start)/x_step)
    ny=int((y_end-y_start)/y_step)
    nz=int((z_end-z_start)/z_step)
    base64_len=int(nx*ny*nz*num_components)
    base64_format='<'+str(base64_len)+'f'

    result=struct.unpack(base64_format, result)
    result=np.array(result).reshape((nx, ny, nz, num_components))
    return result, int(x_start/x_step), int(x_end/x_step), \
    int(y_start/x_step), int(y_end/y_step),\
    int(z_start/z_step), int(z_end/z_step)

'''

import pyJHTDB
from pyJHTDB import libJHTDB

lJHTDB = libJHTDB()
lJHTDB.initialize()
lJHTDB.add_token(token)

def get_frame_big(xstart, ystart, zstart, 
xend, yend, zend, xstep, ystep, zstep,
sim_name, 
timestepstart, timestepend, timestepstep, 
field, num_components):
    r = lJHTDB.getbigCutout(
        data_set=sim_name, fields=field, t_start=timestepstart, 
        t_end=timestepend, t_step=timestepend,
        start=np.array([xstart, ystart, zstart], dtype = np.int),
        end=np.array([xend, yend, zend], dtype = np.int),
        step=np.array([xstep, ystep, zstep], dtype = np.int),
        filter_width=0,
        filename="na")
    return r
'''
def get_full_frame(x_start, x_end, y_start, y_end, z_start, z_end,
sim_name, timestep, field, num_components):
    full = np.zeros((z_end-z_start, y_end-y_start, x_end-x_start, num_components))
    x_len = 1024
    y_len = 1024
    for k in range(z_start, z_end, 1):
        for i in range(x_start, x_end, x_len):
            for j in range(y_start, y_end, y_len):
                x_stop = min(i+x_len, x_end)
                y_stop = min(j+y_len, y_end)
                z_stop = min(k+1, z_end)
                full[k:z_stop,j:y_stop,i:x_stop,:] = get_frame(i,x_stop,x_step, 
                j, y_stop, y_step,
                k, z_stop, z_step,
                sim_name, timestep, field, num_components)[0]
    return full

def download_file(url, file_name):
    try:
        html = requests.get(url, stream=True)
        open(f'{file_name}.json', 'wb').write(html.content)
        return html.status_code
    except requests.exceptions.RequestException as e:
       return e

def get_full_frame_parallel(x_start, x_end, x_step,
y_start, y_end, y_step, 
z_start, z_end, z_step,
sim_name, timestep, field, num_components, num_workers):
    threads= []
    full = np.zeros((int((x_end-x_start)/x_step), 
    int((y_end-y_start)/y_step), 
    int((z_end-z_start)/z_step), num_components), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        done = 0
        x_len = 128
        y_len = 128
        z_len = 128
        for k in range(x_start, x_end, x_len):
            for i in range(y_start, y_end, y_len):
                for j in range(z_start, z_end, z_len):
                    x_stop = min(k+x_len, x_end)
                    y_stop = min(i+y_len, y_end)
                    z_stop = min(j+z_len, z_end)
                    threads.append(executor.submit(get_frame, 
                    k, x_stop, x_step,
                    i, y_stop, y_step,
                    j, z_stop, z_step,
                    sim_name, timestep, field, num_components))
        for task in as_completed(threads):
           r, x1, x2, y1, y2, z1, z2 = task.result()
          
           full[x1-int(x_start/x_step):x2-int(x_start/x_step),
                y1-int(y_start/y_step):y2-int(y_start/y_step),
                z1-int(z_start/z_step):z2-int(z_start/z_step),:] = r.astype(np.float32)
           del r
           done += 1
           print("Done: %i/%i" % (done, len(threads)))
    return full


#name = "channel"
name = "isotropic1024coarse"
#name="mixing"
t0 = time.time()
count = 0
startts = 1
endts = 18
ts_skip = 1
for i in range(startts, endts, ts_skip):
    print("TS %i/%i" % (i, endts))
    f = get_full_frame_parallel(0, 1024, 8,#x
    0, 1024, 8, #y
    0, 1024, 8, #z
    name, i, 
    "u", 3, 
    64)
    print(f.shape)
    f = f.astype(np.float32).swapaxes(0,3).swapaxes(3,2).swapaxes(2,1)
    f_h5 = h5py.File(str(i-1)+'.h5', 'w')
    f_h5.create_dataset("data", data=f)
    f_h5.close()
    count += 1

print("finished")
print(time.time() - t0)
