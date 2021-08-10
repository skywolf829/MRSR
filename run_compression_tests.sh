#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
export PATH="$HOME/sz/bin:$PATH"
#export PATH="$HOME/zfp/bin:$PATH"
#export PATH="$HOME/fpzip/bin:$PATH"
#export PATH="$HOME/tthresh/build:$PATH"

# 2D iso1024 mag
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true 
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bilinear --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true
#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique bicubic --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_4010_psnr_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true

#python3 mixedLOD_octree.py --save_name "NN_mixedLODoctree_SZ" --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion mre --start_metric 0.1 --end_metric 2.0 --metric_skip 0.1 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true --debug true --device cuda:0 
#python3 mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion psnr --start_metric 30 --end_metric 55 --metric_skip 1 --output_folder mag2D_compression --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing false --device cuda:0 --dynamic_downscaling false 
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion psnr --start_metric 28 --end_metric 32 --metric_skip 0.5 --output_folder mag2D_compression_test --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing false --save_TKE true --debug true --device cuda:0 --interpolation_heuristic true --dynamic_downscaling false --preupscaling_PSNR true
#python3 -u mixedLOD_octree.py --save_name NN_biilinearheuristic_SR_octree_SZ --downscaling_technique avgpool2D --upscaling_technique model --model_name SSR_isomag2D --criterion psnr --start_metric 36 --end_metric 36.1 --metric_skip 0.5 --output_folder mag2D_compression_test2 --mode 2D --file isomag2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor sz --load_existing true --save_TKE false --debug false --device cuda:0 --interpolation_heuristic true --save_netcdf true --save_netcdf_octree true --dynamic_downscaling true --preupscaling_PSNR false

#python3 mixedLOD_octree.py --downscaling_technique avgpool2D --upscaling_technique model --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag2D_compression --mode 2D --file 4010.h5 --dims 2 --nx 1024 --ny 1024 --use_compressor true --compressor fpzip --load_existing true --debug true

#python3 -u sz_test.py --metric mre --channels 1 --file isomag2D_compressiontest.h5 --start_value 0.07 --end_value 0.071 --value_skip .002 --dims 2 --nx 1024 --ny 1024 --save_TKE false --output_folder mag2D_compression_test2 --save_netcdf true
#python3 zfp_test.py --metric psnr --channels 1 --file isomag2D_compressiontest.h5 --start_bpv 0.25 --end_bpv 4 --bpv_skip 0.25 --dims 2 --nx 1024 --ny 1024 --output_folder mag2D_compression_pwmre

# 3D iso1024 mag
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_isomag3D --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mag3D_compression --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:1 &
#python3 -u mixedLOD_octree.py --save_name NN_trilinearheuristic_SR_octree_SZ --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso_mag --criterion psnr --start_metric 41.75 --end_metric 42 --metric_skip 0.75 --output_folder mag3D_compression_test --max_LOD 4 --min_chunk 32 --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 1024 --ny 1024 --nz 1024 --save_TKE false --use_compressor true --distributed true --compressor sz --load_existing true --save_netcdf true --save_netcdf_octree true --preupscaling_PSNR false --dynamic_downscaling true --debug false --device cuda:0 --interpolation_heuristic true 
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso_mag --criterion psnr --start_metric 35 --end_metric 55 --metric_skip 0.75 --output_folder mag3D_compression_test --max_LOD 4 --min_chunk 32 --mode 3D --file isomag3D_compressiontest.h5 --dims 3 --nx 1024 --ny 1024 --nz 1024 --save_TKE false --use_compressor true --distributed true --compressor sz --load_existing false --save_netcdf false --preupscaling_PSNR true --dynamic_downscaling false --debug false --device cuda:0 --interpolation_heuristic true 

#python3 -u sz_test.py --metric mre --channels 1 --file isomag3D_compressiontest.h5 --start_value 0.03 --end_value 0.031 --value_skip 0.02 --dims 3 --nx 1024 --ny 1024 --nz 1024 --output_folder mag3D_compression_test --save_TKE false --save_netcdf true

# 3D mixing dataset
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mixing_p --criterion psnr --start_metric 41.5 --end_metric 42 --metric_skip 1.5 --output_folder mixing3D_compression_test --max_LOD 5 --min_chunk 16 --mode 3D --file mixing3D_compressiontest.h5 --dims 3 --nx 512 --ny 512 --nz 512 --use_compressor true --distributed true --compressor sz --load_existing true --save_netcdf true --save_netcdf_octree true --debug true --preupscaling_PSNR false --device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 
#python3 -u sz_test.py --metric mre --channels 1 --file mixing3D_compressiontest2.h5 --start_value 0.005 --end_value 0.95 --value_skip .005 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mixing3D_compression_test2 --save_netcdf false
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_mix3D_LW --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder mix3D_compression --mode 3D --file mixing3D_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:2 &
#python3 sz_test.py --metric psnr --file 1010.h5 --start_value 20 --end_value 80 --value_skip .5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression
#python3 zfp_test.py --metric psnr --file 1010.h5 --start_bpv 0.5 --end_bpv 16 --bpv_skip 0.5 --dims 3 --nx 512 --ny 512 --nz 512 --output_folder mix3D_1010_psnr_compression

# 3D iso1024 VF
#python3 mixedLOD_octree.py --downscaling_technique avgpool3D --upscaling_technique model --model_name SSR_iso3D_VF --criterion psnr --start_metric 20 --end_metric 100 --metric_skip 5 --output_folder iso3D_VF_compression --mode 3D --file iso3D_VF_compressiontest.h5 --dims 3 --nx 128 --ny 128 --nz 128 --use_compressor true --compressor sz --load_existing false --debug true --device cuda:3 &

# Mixing 2D dataset
python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool2D \
--upscaling_technique model --model_name Mixing2D --criterion psnr --start_metric 28 \
--end_metric 60 --metric_skip 1.0 --output_folder Mixing2D_compression_test --max_LOD 7 \
--min_chunk 16 --mode 2D --file Mixing2D_compressiontest.h5 --dims 2 --nx 1024 --ny 1024 \
--use_compressor true --distributed false --compressor sz --load_existing false \
--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

python3 -u sz_test.py --metric mre --channels 1 --file Mixing2D_compressiontest.h5 \
--start_value 0.001 --end_value 0.98 --value_skip .004 --dims 2 --nx 1024 --ny 1024 \
--output_folder Mixing2D_compression_test --save_netcdf false


# Vort dataset
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Vorts --criterion psnr --start_metric 35 \
#--end_metric 60 --metric_skip 100.0 --output_folder Vorts_compression_test --max_LOD 5 \
#--min_chunk 4 --mode 3D --file Vorts_compressiontest.h5 --dims 3 --nx 128 --ny 128 \
#--nz 128 --use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree false --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic false 

#python3 -u sz_test.py --metric mre --channels 1 --file Vorts_compressiontest.h5 \
#--start_value 0.024 --end_value 0.8 --value_skip 1.01 --dims 3 --nx 128 --ny 128 \
#--nz 128 --output_folder Vorts_compression_test --save_netcdf true


# Plume dataset
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Plume --criterion psnr --start_metric 43 \
#--end_metric 60 --metric_skip 100.0 --output_folder Plume_compression_test --max_LOD 5 \
#--min_chunk 4 --mode 3D --file Plume_compressiontest.h5 --dims 3 --nx 512 --ny 128 \
#--nz 128 --use_compressor true --distributed false --compressor sz --load_existing true \
#--save_netcdf true --save_netcdf_octree true --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u sz_test.py --metric mre --channels 1 --file Plume_compressiontest.h5 \
#--start_value 0.027 --end_value 0.960 --value_skip 1.004 --dims 3 --nx 512 --ny 128 \
#--nz 128 --output_folder Plume_compression_test --save_netcdf true

# CombustionVort dataset
#python3 -u mixedLOD_octree.py --save_name NN_SZ --downscaling_technique avgpool3D \
#--upscaling_technique model --model_name Plume --criterion psnr --start_metric 28 \
#--end_metric 60 --metric_skip 1.0 --output_folder Plume_compression_test --max_LOD 5 \
#--min_chunk 4 --mode 3D --file Plume_compressiontest.h5 --dims 3 --nx 128 --ny 768 \
#--nz 512 --use_compressor true --distributed false --compressor sz --load_existing false \
#--save_netcdf false --save_netcdf_octree true --debug true --preupscaling_PSNR true \
#--device cuda:0 --dynamic_downscaling true --interpolation_heuristic true 

#python3 -u sz_test.py --metric mre --channels 1 --file Plume_compressiontest.h5 \
#--start_value 0.003 --end_value 0.07 --value_skip .001 --dims 3 --nx 128 --ny 768 \
#--nz 512 --output_folder Plume_compression_test --save_netcdf false