#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --save_name SSR_GAN1.0_shuffle_nostreamlines --alpha_2 1.0 --train_distributed False --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:0 &
python3 train_spatial_SR.py --save_name SSR_GAN0.5_shuffle_nostreamlines --alpha_2 0.5 --train_distributed False --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:1 &
python3 train_spatial_SR.py --save_name SSR_GAN0.1_shuffle_nostreamlines --alpha_2 0.1 --train_distributed False --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:2 &
python3 train_spatial_SR.py --save_name SSR_CNN_shuffle --train_distributed False --alpha_2 0 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:3 &
python3 train_spatial_SR.py --save_name SSR_GAN0.1_shuffle_streamlines5 --train_distributed False --alpha_6 0.1 --streamline_length 5 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:4 &
python3 train_spatial_SR.py --save_name SSR_GAN0.1_shuffle_streamlines10 --train_distributed False --alpha_6 0.1 --streamline_length 10 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:5 &
python3 train_spatial_SR.py --save_name SSR_GAN0.1_shuffle_streamlines1 --train_distributed False --alpha_6 0.1 --streamline_length 1 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 --device cuda:6 &
python3 train_spatial_SR.py --save_name SSR_GAN0.1_trilin_1gpu --train_distributed False --upsample_mode trilinear --beta_1 0.9 --beta_2 0.999 --device cuda:7 
