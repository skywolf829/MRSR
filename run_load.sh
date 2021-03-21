#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --load_from SSR_isoVF_channelscaling --train_distributed True 
