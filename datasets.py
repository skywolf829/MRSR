import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import zeep
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import struct
import torch

class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')

        self.token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        

        self.subsample_dist = 1
        self.num_items = opt['num_dataset_timesteps'] 

    def get_frame(self,
        x_start, x_end, x_step, 
        y_start, y_end, y_step, 
        z_start, z_end, z_step, 
        sim_name, timestep, field, num_components):
            result=self.client.service.GetAnyCutoutWeb(self.token,sim_name, field, timestep,
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
            result=np.array(result).reshape((nz, ny, nx, num_components))
            return result, int(x_start/x_step), int(x_end/x_step), \
            int(y_start/x_step), int(y_end/y_step),\
            int(z_start/z_step), int(z_end/z_step)

    def get_full_frame_parallel(self,
    x_start, x_end, x_step,
    y_start, y_end, y_step, 
    z_start, z_end, z_step,
    sim_name, timestep, field, num_components, num_workers):
        threads= []
        full = np.zeros((int((z_end-z_start)/z_step), 
        int((y_end-y_start)/y_step), 
        int((x_end-x_start)/x_step), num_components), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            done = 0
            # "Try to limit the number of points in a single query to 2 million"
            # 128^3 is just over 2 million, so we choose that as the maximum
            x_len = 128
            y_len = 128
            z_len = 128
            for k in range(z_start, z_end, z_len):
                for i in range(x_start, x_end, x_len):
                    for j in range(y_start, y_end, y_len):
                        x_stop = min(i+x_len, x_end)
                        y_stop = min(j+y_len, y_end)
                        z_stop = min(k+z_len, z_end)
                        print("adding job")
                        threads.append(executor.submit(self.get_frame, 
                        i,x_stop, x_step,
                        j, y_stop, y_step,
                        k, z_stop, z_step,
                        sim_name, timestep, field, num_components))
            for task in as_completed(threads):
                r, x1, x2, y1, y2, z1, z2 = task.result()
                
                full[z1-z_start:z2-z_start,
                y1-y_start:y2-y_start,
                x1-x_start:x2-x_start,:] = r.astype(np.float32)
                del r
                print("done")
                done += 1
        return full

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist

    def __len__(self):
        return self.num_items - 100

    def resolution(self):
        return self.resolution

    def scale(self, data):
        return data

    def unscale(self, data):
        return data

    def __getitem__(self, index):
        
        x_start = 0
        x_end = self.opt['x_resolution']
        y_start = 0
        y_end = self.opt['y_resolution']
        z_start = 0
        z_end = self.opt['z_resolution']

        if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
            z_start = torch.randint(self.opt['z_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            z_end = min(z_start + self.opt['cropping_resolution']*self.subsample_dist, self.opt['z_resolution'])
        if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
            y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            y_end = min(y_start + self.opt['cropping_resolution']*self.subsample_dist, self.opt['y_resolution'])
        if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
            x_start = torch.randint(self.opt['x_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            x_end = min(x_start + self.opt['cropping_resolution']*self.subsample_dist, self.opt['x_resolution'])
        
        f = self.get_full_frame_parallel(x_start, x_end, self.subsample_dist,#x
        y_start, y_end, self.subsample_dist, #y
        z_start, z_end, self.subsample_dist, #z
        self.opt['dataset_name'], index+100, # skip the first 100 timesteps, duplicates for temporal interpolation
        "u", 3, self.opt['num_networked_workers'])
        
        f = f.astype(np.float32).swapaxes(0,3).swapaxes(3,2).swapaxes(2,1)
        data = torch.tensor(f)

        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = data[:,::-1,:,:]
            if(torch.rand(1).item() > 0.5):
                data = data[:,:,::-1,:]
            if(torch.rand(1).item() > 0.5):
                data = data[:,:,:,::-1]
            
        return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):

            if(opt['load_data_at_start'] or (self.num_items > 0 and \
            (opt['scaling_mode'] == "magnitude" or opt['scaling_mode'] == "channel"))):
                print("Loading " + filename)   
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()

            if(self.num_items == 0):                             
                self.num_channels = d.shape[0]
                self.resolution = d.shape[1:]
                if(self.opt['mode'] == "3Dto2D"):
                    self.resolution = self.resolution[0:len(self.resolution)-1]

            if(opt['load_data_at_start']):
                self.items.append(d)

            if(opt['scaling_mode'] == "magnitude"):  
                mags = torch.norm(d, dim=0)
                m_mag = mags.max()
                if(self.max_mag is None or self.max_mag < m_mag):
                    self.max_mag = m_mag

            if(opt['scaling_mode'] == "channel"):
                for i in range(d.shape[0]):                
                    if(len(self.channel_mins) <= i):
                        self.channel_mins.append(d[i].min())
                        self.channel_maxs.append(d[i].max())
                    else:
                        if(d[i].max() > self.channel_maxs[i]):
                            self.channel_maxs[i] = d[i].max()
                        if(d[i].min() < self.channel_mins[i]):
                            self.channel_mins[i] = d[i].min()
            
            self.num_items += 1

    def __len__(self):
        return self.num_items

    def resolution(self):
        return self.resolution

    def scale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "magnitude"):
            d *= (1/self.max_mag)
        elif (self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[:,i] -= self.channel_mins[i]
                d[:,i] /= (self.channel_maxs[i] - self.channel_mins[i])
                d[:,i] -= 0.5
                d[:,i] *= 2
        return d

    def unscale(self, data):
        d = data.clone()
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                d[:, i] *= 0.5
                d[:, i] += 0.5
                d[:, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[:, i] += self.channel_mins[i]
        elif(self.opt['scaling_mode'] == "magnitude"):
            d *= self.max_mag
        return d

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def __getitem__(self, index):
        if(self.opt['load_data_at_start']):
            data = self.items[index]
        else:
            print("trying to load " + str(index) + ".h5")
            f = h5py.File(os.path.join(self.opt['data_folder'], str(index)+".h5"), 'r')
            print("converting " + str(index) + ".h5 to tensor")
            data =  torch.tensor(f.get('data'))
            f.close()
            print("converted " + str(index) + ".h5 to tensor")

        '''
        if(self.opt['scaling_mode'] == "channel"):
            for i in range(self.num_channels):
                data[i] -= self.channel_mins[i]
                data[i] *= (1 / (self.channel_maxs[i] - self.channel_mins[i]))
                data[i] -= 0.5
                data[i] *= 2
        elif(self.opt['scaling_mode'] == "magnitude"):
            data *= (1 / self.max_mag)
        '''

        if(self.opt['mode'] == "3Dto2D"):
            data = data[:,:,:,int(data.shape[3]/2)]

        #data = np2torch(data, "cpu")
        #print("returning " + str(index) + " data")
        
        return data