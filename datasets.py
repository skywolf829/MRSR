import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import zeep
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import struct
import torch
import h5py
from utility_functions import AvgPool3D, AvgPool2D
import torch.nn.functional as F

class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
        self.token="edu.osu.buckeyemail.wurster.18-92fb557b"
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
                        
            self.client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
            result=self.client.service.GetAnyCutoutWeb(self.token,
                sim_name, field, timestep,
                x_start+1, 
                y_start+1, 
                z_start+1, 
                x_end, y_end, z_end,
                x_step, y_step, z_step, 0, "")  # put empty string for the last parameter
            # transfer base64 format to numpy
            nx=int((x_end-x_start)/x_step)
            ny=int((y_end-y_start)/y_step)
            nz=int((z_end-z_start)/z_step)
            base64_len=int(nx*ny*nz*num_components)
            base64_format='<'+str(base64_len)+'f'

            result=struct.unpack(base64_format, result)
            result=np.array(result).reshape((nz, ny, nx, num_components)).swapaxes(0,2)
            return result, x_start, x_end, y_start, y_end, z_start, z_end

    def get_full_frame_parallel(self,
    x_start, x_end, x_step,
    y_start, y_end, y_step, 
    z_start, z_end, z_step,
    sim_name, timestep, field, num_components, num_workers):
        threads= []
        full = np.zeros((int((x_end-x_start)/x_step), 
        int((y_end-y_start)/y_step), 
        int((z_end-z_start)/z_step), num_components), dtype=np.float32)
        
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            done = 0
            # "Try to limit the number of points in a single query to 2 million"
            # 128^3 is just over 2 million, so we choose that as the maximum
            x_len = 128
            y_len = 128
            z_len = 128
            for k in range(x_start, x_end, x_len):
                for i in range(y_start, y_end, y_len):
                    for j in range(z_start, z_end, z_len):
                        x_stop = min(k+x_len, x_end)
                        y_stop = min(i+y_len, y_end)
                        z_stop = min(j+z_len, z_end)
                        threads.append(executor.submit(self.get_frame, 
                        k, x_stop, x_step,
                        i, y_stop, y_step,
                        j, z_stop, z_step,
                        sim_name, timestep, field, num_components))
            for task in as_completed(threads):
                r, x1, x2, y1, y2, z1, z2 = task.result()
                x1 -= x_start
                x2 -= x_start
                y1 -= y_start
                y2 -= y_start
                z1 -= z_start
                z2 -= z_start
                x1 = int(x1 / x_step)
                x2 = int(x2 / x_step)
                y1 = int(y1 / y_step)
                y2 = int(y2 / y_step)
                z1 = int(z1 / z_step)
                z2 = int(z2 / z_step)
                full[x1:x2,y1:y2,z1:z2,:] = r.astype(np.float32)
                del r
                done += 1
                #print("Done: %i/%i" % (done, len(threads)))
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
            z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist
        if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
            y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
        if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
            x_start = torch.randint(self.opt['x_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist

        
        f = self.get_full_frame_parallel(x_start, x_end, self.subsample_dist,#x
        y_start, y_end, self.subsample_dist, #y
        z_start, z_end, self.subsample_dist, #z
        self.opt['dataset_name'], index*self.opt["ts_skip"], # skip the first 100 timesteps, duplicates for temporal interpolation
        "u", 3, self.opt['num_networked_workers'])
        '''
        f, _, _, _, _, _, _ = self.get_frame(x_start, x_end, self.subsample_dist, 
        y_start, y_end, self.subsample_dist, 
        z_start, z_end, self.subsample_dist, 
        self.opt['dataset_name'], index+100, "u", 3)
        '''

        f = f.astype(np.float32).swapaxes(0,3).swapaxes(3,2).swapaxes(2,1)
        data = torch.tensor(f)

        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[3])
            
        return data

class LocalTemporalDataset(torch.utils.data.Dataset):
    
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        self.item_names = []
        self.subsample_dist = 1
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):
            self.item_names.append(filename)

            if(opt['load_data_at_start'] or (self.num_items > 0 and \
            (opt['scaling_mode'] == "magnitude" or opt['scaling_mode'] == "channel"))):
                print("Loading " + filename)   
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()

            if(self.num_items == 0):           
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()                  
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
        return self.num_items - self.opt['training_seq_length'] + 1

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

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist
        
    def __getitem__(self, index):
        if(self.opt['load_data_at_start']):
            data = self.items[index]
        else:

            #print("trying to load " + str(self.item_names[index]) + ".h5")
            x_start = 0
            x_end = self.opt['x_resolution']
            y_start = 0
            y_end = self.opt['y_resolution']
            z_start = 0
            z_end = self.opt['z_resolution']

            if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                z_start = torch.randint(self.opt['z_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist
            if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
                y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
            if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
                x_start = torch.randint(self.opt['x_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist


            
            
            

            #print("converting " + self.item_names[index] + " to tensor")
            all_frames = []
            for i in range(self.opt['training_seq_length']):
                f = h5py.File(os.path.join(self.opt['data_folder'], self.item_names[index+i]), 'r')
                data =  torch.tensor(f['data'][:,x_start:x_end,
                    y_start:y_end,
                    z_start:z_end])
                f.close()
                if(self.subsample_dist > 1):
                    data = AvgPool3D(data.unsqueeze(0), self.subsample_dist)[0]
                all_frames.append(data)
            data = torch.stack(all_frames, dim=0)
            #print("converted " + self.item_names[index] + ".h5 to tensor")

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
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[3])

        return (data[0:1], data[self.opt['training_seq_length']-1:self.opt['training_seq_length']], 
        data[1:self.opt['training_seq_length']-1], (index, index+self.opt['training_seq_length']-1))

class TestingDataset(torch.utils.data.Dataset):
    def __init__(self, location):
        self.location = location
        print("Initializing dataset")
        self.ext = ""
        self.item_names = []
        for filename in os.listdir(location):
            self.item_names.append(filename.split(".")[0])
            self.ext = filename.split(".")[1]
        self.item_names.sort(key=int)
        print("Dataset has " + str(len(self.item_names)) + " items")
    def __len__(self):
        return len(self.item_names)
    def __getitem__(self, index):       
        print("Loading " + str(index))
        f = h5py.File(os.path.join(self.location, self.item_names[index]+"."+self.ext), 'r')
        a = torch.Tensor(f['data'])
        f.close()
        return a.unsqueeze(0)

class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.channel_mins = []
        self.channel_maxs = []
        self.max_mag = None
        self.num_items = 0
        self.items = []
        self.item_names = []
        self.subsample_dist = 1
        print("Initializing dataset")
        for filename in os.listdir(self.opt['data_folder']):

            self.item_names.append(filename)

            
            if(opt['load_data_at_start'] or (self.num_items > 0 and \
            (opt['scaling_mode'] == "magnitude" or opt['scaling_mode'] == "channel"))):
                print("Loading " + filename)   
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()

            if(self.num_items == 0):           
                f = h5py.File(os.path.join(self.opt['data_folder'], filename), 'r')
                d = torch.tensor(f.get('data'))
                f.close()                  
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
        self.item_names.sort()

        if(opt['training_data_amount'] < 1.0):
            end = int(opt['training_data_amount'] * len(self.item_names))
            import random
            random.seed(0)
            random.shuffle(self.item_names)
            while(len(self.item_names) > end):
                self.item_names.pop(len(self.item_names)-1)
                self.num_items -= 1
        if(opt['coarse_training'] > 2):
            self.item_names = self.item_names[::opt['coarse_training']]
            self.num_items = len(self.item_names)
    

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

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist
        
    def __getitem__(self, index):
        if(self.opt['load_data_at_start']):
            data = self.items[index]
        else:

            #print("trying to load " + str(self.item_names[index]) + ".h5")
            f = h5py.File(os.path.join(self.opt['data_folder'], self.item_names[index]), 'r')
            x_start = 0
            x_end = self.opt['x_resolution']
            y_start = 0
            y_end = self.opt['y_resolution']
            if(self.opt['mode'] == "3D"):
                z_start = 0
                z_end = self.opt['z_resolution']
                if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                    z_start = torch.randint(self.opt['z_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                    z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

            if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
                y_start = torch.randint(self.opt['y_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
            if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
                x_start = torch.randint(self.opt['x_resolution'] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist
            
            if(self.opt['downsample_mode'] == "average_pooling"):
                #print("converting " + self.item_names[index] + " to tensor")
                if(self.opt['mode'] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end,
                        z_start:z_end])
                elif(self.opt['mode'] == "2D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end])
                f.close()
                
                if(self.subsample_dist > 1):
                    if(self.opt["mode"] == "3D"):
                        data = AvgPool3D(data.unsqueeze(0), self.subsample_dist)[0]
                    elif(self.opt['mode'] == "2D"):
                        data = AvgPool2D(data.unsqueeze(0), self.subsample_dist)[0]
                    
            elif(self.opt['downsample_mode'] == "subsampling"):
                if(self.opt["mode"] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist,
                        z_start:z_end:self.subsample_dist])
                elif(self.opt['mode'] == "2D"):       
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist])                 
                f.close()
            else:
                if(self.opt["mode"] == "3D"):
                    data =  torch.tensor(f['data'][:,x_start:x_end,
                        y_start:y_end,
                        z_start:z_end])
                elif(self.opt['mode'] == "2D"):   
                    data =  torch.tensor(f['data'][:,x_start:x_end:self.subsample_dist,
                        y_start:y_end:self.subsample_dist])
                f.close()
                data = F.interpolate(data.unsqueeze(0), scaling_factor=float(1/self.subsample_dist), 
                mode=self.opt['downsample_mode'], align_corners=True)[0]
            #print("converted " + self.item_names[index] + ".h5 to tensor")


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
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data = torch.flip(data,[3])

        return data