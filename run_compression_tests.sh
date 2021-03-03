#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
python3 mixedLOD_octree.py --upscaling_technique bilinear --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
python3 mixedLOD_octree.py --upscaling_technique bicubic --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
python3 mixedLOD_octree.py --upscaling_technique model --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
python3 sz_test.py --metric mre --start_value 0.001 --end_value 0.1 --value_skip 0.01 --output_folder mag2D_4010_mre