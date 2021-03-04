#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
#python3 mixedLOD_octree.py --upscaling_technique bilinear --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
#python3 mixedLOD_octree.py --upscaling_technique bicubic --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
#python3 mixedLOD_octree.py --upscaling_technique model --criterion mre --start_metric 0.001 --end_metric 0.1 --metric_skip 0.01 --output_folder mag2D_4010_mre
#python3 sz_test.py --metric mre --start_value 0.001 --end_value 0.1 --value_skip 0.01 --output_folder mag2D_4010_mre


#python3 mixedLOD_octree.py --folder mix_p --file 1010.h5 --downscaling_technique avgpool3D --upscaling_technique trilinear --criterion psnr --start_metric 10 --end_metric 100 --metric_skip 10 --output_folder mixing3D_1010_psnr --mode 3D --min_chunk 16 --debug false
python3 mixedLOD_octree.py --folder mix_p --file 1010.h5 --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --criterion psnr --start_metric 10 --end_metric 100 --metric_skip 10 --output_folder mixing3D_1010_psnr --mode 3D --min_chunk 16 --debug true --distributed true
python3 sz_test.py --folder mix_p --file 1010.h5 --metric psnr --start_value 10 --end_value 100 --value_skip 10 --output_folder mixing3D_1010_psnr --dims 3 --nx 512 --ny 512 --nx 512