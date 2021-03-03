#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 mixedLOD_octree.py --upscaling_technique bilinear
python3 mixedLOD_octree.py --upscaling_technique bicubic
python3 mixedLOD_octree.py --upscaling_technique model
python3 sz_test.py