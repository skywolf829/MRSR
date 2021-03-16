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

#python3 mixedLOD_octree.py --save_name "NN_mixedLODoctree_SZ" --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion mre --start_metric 0.1 --end_metric 2.0 --metric_skip 0.1 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true --debug true --device cuda:0 
#python3 mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion psnr --start_metric 30 --end_metric 55 --metric_skip 1 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing false --device cuda:0 --dynamic_downscaling false 
python3 -u mixedLOD_octree.py --save_name NN_bilinearheuristic_mixedLOD_octree_SZ --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion psnr --start_metric 30 --end_metric 55 --metric_skip 0.5 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true --save_TKE true --debug true --device cuda:0 --interpolation_heuristic true 

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true --debug true

python3 -u sz_test.py --metric mre --channels 1 --file isomag2D_compressiontest.h5 --start_value 0.005 --end_value 0.05 --value_skip .005 --dims 2 --nx 1024 --ny 1024 --save_TKE true --output_folder mag2D_compression
#python3 zfp_test.py --metric psnr --channels 1 --file isomag2D_compressiontest.h5 --start_bpv 0.25 --end_bpv 4 --bpv_skip 0.25 --dims 2 --nx 1024 --ny 1024 --output_folder mag2D_compression_pwmre

# 3D iso1024 mag
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_isomag3D --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag3D_compression --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:1 &
#python3 -u mixedLOD_octree.py --save_name NN_trilinearheuristic_mixedLOD_octree_SZ --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso_mag --criterion psnr --start_metric 29 --end_metric 55 --metric_skip 0.5 --output_folder mag3D_compression --max_LOD 4 --min_chunk 32 --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 1024 --ny 1024 --nz 1024 --use_compressor true --distributed true --compressor sz --load_existing false --debug false --device cuda:0 --interpolation_heuristic true 
#python3 -u sz_test.py --metric psnr --channels 1 --file isomag3D_compressiontest.h5 --start_value 1 --end_value 56 --value_skip 3 --dims 3 --nx 1024 --ny 1024 --nz 1024 --output_folder mag3D_compression

# 3D mixing dataset
#python3 -u mixedLOD_octree.py --save_name NN_trilinearheuristic_mixedLOD_octree_SZ --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --criterion psnr --start_metric 36.5 --end_metric 55 --metric_skip 100 --output_folder mixing3D_compression --mode 3D --file mixing3D_compressiontest.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --distributed true --compressor sz --load_existing true --save_netcdf true --debug false --device cuda:0 --interpolation_heuristic true 
#python3 -u sz_test.py --metric psnr --channels 1 --file mixing3D_compressiontest.h5 --start_value 7 --end_value 55 --value_skip 100 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mixing3D_compression --save_netcdf true
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mix3D_LW --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_compression --mode 3D --file mixing3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:2 &
#python3 sz_test.py --metric psnr --file 1010.h5 --start_value 20 --end_value 80 --value_skip .5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression
#python3 zfp_test.py --metric psnr --file 1010.h5 --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression

# 3D iso1024 VF
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso3D_VF --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder iso3D_VF_compression --mode 3D --file iso3D_VF_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:3 &
