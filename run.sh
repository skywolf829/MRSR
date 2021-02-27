#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --save_name SSR_isomag2D --train_distributed False --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/isomag2D 
#python train_spatial_SR.py --save_name SSR_isomag2D --train_distributed False --upsample_mode shuffle --beta_1 0.9 --num_workers 6 --beta_2 0.999 --data_folder InputData/isomag2D 
