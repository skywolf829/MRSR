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

def MSE(x, GT):
    return ((x-GT)**2).mean()

def PSNR(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    p = 20 * np.log10(max_diff) - 10*np.log10(MSE(x, GT))
    return p

def relative_error(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    val = np.abs(GT-x).max() / max_diff
    return val

def pw_relative_error(x, GT):
    val = np.abs(GT-x) / GT
    return val.max()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="4010.h5",type=str,help='File to test compression on')
    parser.add_argument('--folder',default="quadtree_images",type=str,help='File to test compression on')
    parser.add_argument('--dims',default=2,type=int,help='# dimensions')
    parser.add_argument('--nx',default=1024,type=int,help='# x dimension')
    parser.add_argument('--ny',default=1024,type=int,help='# y dimension')
    parser.add_argument('--nz',default=1024,type=int,help='# z dimension')
    parser.add_argument('--output_folder',default="mag2D_4010",type=str,help='Where to save results')
    parser.add_argument('--start_bpv',default=0.5,type=float,help='bits per value to start tests at')
    parser.add_argument('--end_bpv',default=16,type=float,help='bits per value to end tests at')
    parser.add_argument('--bpv_skip',default=0.5,type=float,help='bits per value increment by')
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
    results['rec_psnr'] = []
    results['compression_time'] = []
    f = h5py.File(os.path.join(input_folder, args['file']), "r")
    d = np.array(f['data'][0])
    f.close()
    d.tofile(args['file'] + ".dat")
    value = args['start_value']
    while(value < args['end_value']):
        command = "zfp -f -i " + args['file'] + ".dat -z " + \
            args['file']+".dat.zfp -o " + \
            args['file']+".dat.zfp.out -" + str(args['dims']) + " " + \
            str(args['nx']) + " " + str(args['ny'])
        if(args['dims'] == 3):
            command = command + " " + str(args['nz'])
        command = command + " -r " + str(value)
        start_t = time.time()
        print("Running: " + command)
        os.system(command)
        compression_time = time.time() - start_t

        f_size_kb = os.path.getsize(args['file'] + ".dat.zfp") / 1024

        command = "zfp -f -z " + args['file'] + ".dat.zfp -" + str(args['dims']) + " " + \
            str(args['nx']) + " " + str(args['ny'])
        if(args['dims'] == 3):
            command = command + " " + str(args['nz'])
        command = command + " -S " + str(value) 

        os.system(command)

        dc = np.fromfile(args['file']+".dat.zfp.out")
        dc.dtype = np.float32
        if(args['dims'] == 2):
            dc = dc.reshape(args['nx'], args['ny'])
        elif(args['dims'] == 3):
            dc = dc.reshape(args['nx'], args['ny'], args['nz'])    

        rec_psnr = PSNR(dc,d)
                
        if(args['dims'] == 2):
            im = dc - dc.min()
            im *= (255/dc.max())
            im = im.astype(np.uint8)
        elif(args['dims'] == 3):      
            im = dc - dc.min()
            im *= (255/dc.max())
            im = im.astype(np.uint8)[:,:,int(im.shape[2]/2)]
        
        imageio.imwrite(os.path.join(save_folder, "zfp_"+args['file']+"_"+str(value)+".png"), im)

        command = "mv " + args['file']+".dat.zfp " + save_folder +"/psnr_"+str(value)+"_"+args['file']+".zfp"
        os.system(command)

        results['psnrs'].append(value)
        results['file_size'].append(f_size_kb)
        results['compression_time'].append(compression_time)
        results['rec_psnr'].append(rec_psnr)
        value += args['value_skip']

    if(os.path.exists(os.path.join(save_folder, "results.pkl"))):
        all_data = load_obj(os.path.join(save_folder, "results.pkl"))
    else:
        all_data = {}

    all_data['zfp'] = results
    save_obj(all_data, os.path.join(save_folder, "results.pkl"))

    os.remove(args['file']+'.dat')    
    #os.remove(args['file']+'.dat.zfp')
    os.remove(args['file']+'.dat.zfp.out')