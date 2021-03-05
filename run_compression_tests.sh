#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_sz --sz_compress true --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --sz_mode 2 --debug true
python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_sz --sz_compress true --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --sz_mode 2 --debug true
python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_sz --sz_compress true --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --sz_mode 2 --debug true --model_name SSR_isomag2D
python3 sz_test.py --metric psnr --start_value 20 --end_value 100 --value_skip 5 --output_folder mag2D_4010_psnr_sz


#python3 mixedLOD_octree.py --folder mix_p --file 1010.h5 --downscaling_technique avgpool3D --upscaling_technique trilinear --criterion psnr --start_metric 10 --end_metric 100 --metric_skip 10 --output_folder mixing3D_1010_psnr --mode 3D --min_chunk 16 --debug false
#python3 mixedLOD_octree.py --folder mix_p --file 1010.h5 --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --criterion psnr --start_metric 10 --end_metric 100 --metric_skip 10 --output_folder mixing3D_1010_psnr --mode 3D --min_chunk 16 --debug true --distributed true
#python3 sz_test.py --folder mix_p --file 1010.h5 --metric psnr --start_value 10 --end_value 100 --value_skip 10 --output_folder mixing3D_1010_psnr --dims 3 --nx 512 --ny 512 --nz 512