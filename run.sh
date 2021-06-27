#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/MRSR
#cd ~/MRSR

#python3 -u train_spatial_SR.py --save_name Isomag2D --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 --num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024
#python3 train_spatial_SR.py --save_name SSR_mixing3D --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/mix_p --mode 3D --patch_size 96 --training_patch_size 96 --num_blocks 3 --base_num_kernels 96 --x_resolution 512 --y_resolution 512 --z_resolution 512 --epochs 50
#python3 -u train_spatial_SR.py --save_name SSR_isoVF_channelscaling --scaling_mode channel --alpha_1 0 --alpha_2 0.1 --alpha_4 1 --alpha_6 0.1 --streamline_length 1 --streamline_res 100 --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --num_channels 3 --data_folder InputData/isoVF3D --mode 3D --epochs 1 --patch_size 96 --training_patch_size 96 --num_blocks 3 --base_num_kernels 96 --min_dimension_size 16 --x_resolution 128 --y_resolution 128 --z_resolution 128
#python3 train_spatial_SR.py --save_name SSR_isomag3D_LW --train_distributed True --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/iso_mag --mode 3D --patch_size 128 --training_patch_size 128 --num_blocks 1 --base_num_kernels 32 --min_dimension_size 32 --x_resolution 512 --y_resolution 512 --z_resolution 512 --epochs 7

#python3 -u train_spatial_SR.py --save_name Isomag2D_100percent --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
#--num_workers 0 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024 --training_data_amount 1.0 --epochs 25 \
#--cropping_resolution 512

#python3 -u train_spatial_SR.py --save_name Isomag2D_75percent --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
#--num_workers 0 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024 --training_data_amount 0.75 --epochs 50 \
#--cropping_resolution 512

#python3 -u train_spatial_SR.py --save_name Isomag2D_50percent --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
#--num_workers 0 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024 --training_data_amount 0.5 --epochs 100 \
#--cropping_resolution 512

#python3 -u train_spatial_SR.py --save_name Isomag2D_25percent --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
#--num_workers 0 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024 --training_data_amount 0.25 --epochs 200 \
#--cropping_resolution 512

#python3 -u train_spatial_SR.py --save_name Isomag2D_2percent --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
#--num_workers 0 --beta_2 0.999 --data_folder TrainingData/Isomag2D --mode 2D --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --base_num_kernels 96 --x_resolution 1024 --y_resolution 1024 --training_data_amount 0.02 --epochs 2500 \
#--cropping_resolution 512

python3 -u train_spatial_SR.py --save_name Supernova --train_distributed True --gpus_per_node 4 \
--num_workers 0 --data_folder TrainingData/Supernova --mode 3D --patch_size 96 --training_patch_size 96 \
--x_resolution 448 --y_resolution 448 --z_resolution 448 --epochs 10 --min_dimension_size 28 \
--cropping_resolution 96