#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 train_temporal_SR.py --save_name TSRTVD --train_distributed True --beta_1 0.0 --beta_2 0.999