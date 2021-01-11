import numpy as np
import h5py 
import os

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(MVTVSSR_folder_path):
    if(".npy" in filename):
        print("Removing " + filename)
        f = np.load(os.path.join(MVTVSSR_folder_path, filename))
        f_name = filename.split('.')[0]
        f_name = f_name + ".h5"
        hf = h5py.File(f_name, 'w')
        hf.create_dataset('data',data=f)
        hf.close()
        print("Created " + f_name)
        os.remove(os.path.join(MVTVSSR_folder_path, filename))
        print("Deleted + " os.path.join(MVTVSSR_folder_path, filename))
    else:
        print("Skipping " + filename)