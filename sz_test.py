import os
import numpy as np
import imageio
import argparse
import h5py
import time
import pickle

def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--folder',default="quadtree_images",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--output_folder',default="mag2D_4010",type=str,help='Where to save results')
    parser.add_argument('--start_value',default=10,type=float,help='PSNR to start tests at')
    parser.add_argument('--end_value',default=100,type=float,help='PSNR to end tests at')
    parser.add_argument('--value_skip',default=10,type=float,help='PSNR increment by')
    parser.add_argument('--metric',default='psnr',type=str)
    

    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "TestingData", args['folder'])
    
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['output_folder'])

    if(not os.path.exists(save_folder)):
        os.makedirs(save_folder)
    
    # psnr -> metrics
    results = {}
    results['file_size'] = []
    results['psnrs'] = []
    results['compression_time'] = []
    f = h5py.File(os.path.join(input_folder, args['file']), "r")
    d = np.array(f['data'][0])
    f.close()
    d.tofile(args['file'] + ".dat")
    value = args['start_value']
    while(value < args['end_value']):
        command = "sz -z -f -i " + args['file'] + ".dat -" + str(args['dims']) + " " + \
            str(args['nx']) + " " + str(args['ny'])
        if(args['dims'] == 3):
            command = command + " " + str(args['nz'])
        if(args['metric'] == "psnr"):
            command = command + " -S " + str(value)
        elif(args['metric'] == "mre"):
            command = command + " -R " + str(value)
        elif(args['metric'] == "pw_mre"):
            command = command + " -P " + str(value)
        start_t = time.time()
        os.system(command)
        compression_time = time.time() - start_t

        f_size_kb = os.path.getsize(args['file'] + ".dat.sz") / 1024

        command = "sz -x -f -s " + args['file'] + ".dat.sz -" + str(args['dims']) + " " + \
            str(args['nx']) + " " + str(args['ny'])
        if(args['dims'] == 3):
            command = command + " " + str(args['nz'])
        command = command + " -S " + str(value) 

        os.system(command)

        dc = np.fromfile(args['file']+".dat.sz.out")
        dc.dtype = np.float32
        if(args['dims'] == 2):
            dc = dc.reshape(args['nx'], args['ny'])
            im = dc - dc.min()
            im *= (255/dc.max())
            im = im.astype(np.uint8)
        elif(args['dims'] == 3):
            dc = dc.reshape(args['nx'], args['ny'], args['nz'])            
            im = dc - dc.min()
            im *= (255/dc.max())
            im = im.astype(np.uint8)[:,:,int(im.shape[2]/2)]
        imageio.imwrite(os.path.join(save_folder, "sz_"+args['file']+"_"+str(value)+".png"), im)

        command = "mv " + args['file']+".dat.sz " + save_folder +"/psnr_"+str(value)+"_"+args['file']+".sz"
        os.system(command)

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        value += args['value_skip']

    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
    else:
        all_data = {}

    all_data['sz'] = results
    save_obj(all_data, os.path.join(save_folder, "results.pkl"))

    os.remove(args['file']+'.dat')    
    #os.remove(args['file']+'.dat.sz')
    os.remove(args['file']+'.dat.sz.out')