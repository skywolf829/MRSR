#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
export PATH="$HOME/zfp/bin:$PATH"

export PATH="$HOME/fpzip/bin:$PATH"


# 2D iso1024 mag
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true 
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true --debug true
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true --debug true

#python3 sz_test.py --metric psnr --start_value 20 --end_value 100 --value_skip 5 --output_folder mag2D_4010_psnr_compression
#python3 zfp_test.py --metric psnr --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --output_folder mag2D_4010_psnr_compression

# 3D mixing dataset
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique trilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_1010_psnr_compression --mode 3D --file 1010.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --compressor sz
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique trilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_1010_psnr_compression --mode 3D --file 1010.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --compressor fpzip

python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --distributed true --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_1010_psnr_compression --mode 3D --file 1010.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --compressor sz --debug true
python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --distributed true --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_1010_psnr_compression --mode 3D --file 1010.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --compressor fpzip --debug true
 
python3 sz_test.py --metric psnr --file 1010.h5 --start_value 20 --end_value 100 --value_skip 5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression
python3 zfp_test.py --metric psnr --file 1010.h5 --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression
