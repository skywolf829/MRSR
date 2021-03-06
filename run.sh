#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --save_name SSR_isomag2D_LW --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 --num_blocks 1 --base_num_kernels 32 --x_resolution 1024 --y_resolution 1024
#python3 train_spatial_SR.py --save_name SSR_mix3D_LW --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/mix_p --mode 3D --patch_size 128 --training_patch_size 128 --num_blocks 1 --base_num_kernels 32 --x_resolution 512 --y_resolution 512 --z_resolution 512
