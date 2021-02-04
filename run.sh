#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --save_name SSR --train_distributed True --alpha_6 0.1 --streamline_length 5 --upsample_mode shuffle --beta_1 0.9 --beta_2 0.999 
