#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_spatial_SR.py --save_name SSR --device cuda:0 &
python3 train_temporal_SR.py --save_name TSRTVD --device cuda:1