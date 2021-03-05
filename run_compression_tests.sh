#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
export PATH="$HOME/zfp/bin:$PATH"

export PATH="$HOME/fpzip/bin:$PATH"



#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor zfp
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip

python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor zfp
python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor zfp
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip

python3 sz_test.py --metric psnr --start_value 20 --end_value 100 --value_skip 5 --output_folder mag2D_4010_psnr_compression
python3 zfp_test.py --metric psnr --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --output_folder mag2D_4010_psnr_compression
