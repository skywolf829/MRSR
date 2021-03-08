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

python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D_1blocks_96kernels --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:0 &
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true --debug true

python3 sz_test.py --metric psnr --channels 1 --file isomag2D_compressiontest.h5 --start_value 20 --end_value 100 --value_skip 5 --dims 1 --nx 1024 --ny 1024 --output_folder mag2D_compression
#python3 zfp_test.py --metric psnr --file isomag2D_compressiontest.h5 --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --dims 1 --nx 1024 --ny 1024 --output_folder mag2D_compression

# 3D iso1024 mag
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_isomag3D --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag3D_compression --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:1 &

# 3D mixing dataset
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mix3D_LW --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_compression --mode 3D --file mixing3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:2 &
#python3 sz_test.py --metric psnr --file 1010.h5 --start_value 20 --end_value 100 --value_skip 5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression
#python3 zfp_test.py --metric psnr --file 1010.h5 --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression

# 3D iso1024 VF
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso3D_VF --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder iso3D_VF_compression --mode 3D --file iso3D_VF_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:3 &
