#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_temporal_SR.py --save_name TSRTVD --train_distributed False --beta_1 0.0 --beta_2 0.999 --temporal_model TSRTVD --device cuda:0 &
python3 train_temporal_SR.py --save_name T_UNet --train_distributed False --beta_1 0.5 --beta_2 0.999 --temporal_model UNet --device cuda:1 &
python3 train_spatial_SR.py --save_name SSR_shuffle_1gpu --train_distributed False --upsample_mode shuffle --beta_1 0.9 --beta_2 0.0 --device cuda:2 &
python3 train_spatial_SR.py --save_name SSR_shuffle_CNN --train_distributed False --alpha_2 0 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.0 --device cuda:3 &
python3 train_spatial_SR.py --save_name SSR_shuffle_streamlines5 --train_distributed False --alpha_6 0.1 --streamline_length 5 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.0 --device cuda:4 &
python3 train_spatial_SR.py --save_name SSR_shuffle_streamlines5 --train_distributed False --alpha_6 0.1 --streamline_length 10 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.0 --device cuda:5 &
python3 train_spatial_SR.py --save_name SSR_shuffle_streamlines5 --train_distributed False --alpha_6 0.1 --streamline_length 1 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.0 --device cuda:6