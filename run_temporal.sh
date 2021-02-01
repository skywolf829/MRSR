#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_temporal_SR.py --save_name TSRTVD --train_distributed False --beta_1 0.0 --beta_2 0.999 --temporal_model TSRTVD --device 0 &
python3 train_temporal_SR.py --save_name T_UNet --train_distributed False --beta_1 0.5 --beta_2 0.999 --temporal_model UNet --device 1 