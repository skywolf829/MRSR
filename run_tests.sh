#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/MRSR

#python3 test_SSR.py --output_file_name iso2D_2x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 2 --testing_method bilinear --print True --device cuda:1 --data_folder isomag2D --mode 2D 
#python3 test_SSR.py --output_file_name iso2D_2x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 2 --testing_method bicubic --print True --device cuda:2 --data_folder isomag2D --mode 2D 
#python3 test_SSR.py --output_file_name iso2D_2x --full_resolution 1024 --channels 1 --save_name model --scale_factor 2 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_4x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 4 --testing_method bilinear --model_name Isomag2D --print True --device cuda:1 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_4x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 4 --testing_method bicubic --model_name Isomag2D --print True --device cuda:2 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_4x --full_resolution 1024 --channels 1 --save_name model --scale_factor 4 --testing_method model --model_name Isomag2D --print True --device cuda:0 --data_folder Isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_8x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 8 --testing_method bilinear --model_name Isomag2D --print True --device cuda:1 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_8x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 8 --testing_method bicubic --model_name Isomag2D --print True --device cuda:2 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_8x --full_resolution 1024 --channels 1 --save_name model --scale_factor 8 --testing_method model --model_name Isomag2D --print True --device cuda:0 --data_folder Isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_16x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 16 --testing_method bilinear --model_name Isomag2D --print True --device cuda:1 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_16x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 16 --testing_method bicubic --model_name Isomag2D --print True --device cuda:2 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_16x --full_resolution 1024 --channels 1 --save_name model --scale_factor 16 --testing_method model --model_name Isomag2D --print True --device cuda:0 --data_folder Isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_32x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 32 --testing_method bilinear --model_name Isomag2D --print True --device cuda:1 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_32x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 32 --testing_method bicubic --model_name Isomag2D --print True --device cuda:2 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_32x --full_resolution 1024 --channels 1 --save_name model --scale_factor 32 --testing_method model --model_name Isomag2D --print True --device cuda:0 --data_folder Isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_64x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 64 --testing_method bilinear --model_name Isomag2D --print True --device cuda:1 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_64x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 64 --testing_method bicubic --model_name Isomag2D --print True --device cuda:2 --data_folder Isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_64x --full_resolution 1024 --channels 1 --save_name model --scale_factor 64 --testing_method model --model_name Isomag2D --print True --device cuda:0 --data_folder Isomag2D --mode 2D 
#############################################

#python3 test_SSR.py --output_file_name mixing3D_2x_new --full_resolution 512 --channels 1 --save_name trilinear --scale_factor 2 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_2x_new --full_resolution 512 --channels 1 --save_name model --scale_factor 2 --testing_method model --model_name SSR_mixing3D --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_4x_new --full_resolution 512 --channels 1 --save_name trilinear --scale_factor 4 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_4x_new --full_resolution 512 --channels 1 --save_name model --scale_factor 4 --testing_method model --model_name SSR_mixing3D --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_8x_new --full_resolution 512 --channels 1 --save_name trilinear --scale_factor 8 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_8x_new --full_resolution 512 --channels 1 --save_name model --scale_factor 8 --testing_method model --model_name SSR_mixing3D --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_16x_new --full_resolution 512 --channels 1 --save_name trilinear --scale_factor 16 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_16x_new --full_resolution 512 --channels 1 --save_name model --scale_factor 16 --testing_method model --model_name SSR_mixing3D --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_32x_new --full_resolution 512 --channels 1 --save_name trilinear --scale_factor 32 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_32x_new --full_resolution 512 --channels 1 --save_name model --scale_factor 32 --testing_method model --model_name SSR_mixing3D --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 

###############################################

#python3 test_SSR.py --output_file_name iso3D_2x --full_resolution 1024 --channels 1 --save_name trilinear --scale_factor 2 --testing_method trilinear --print True --device cuda:0 --data_folder iso1024mag --mode 3D 
#python3 test_SSR.py --output_file_name iso3D_2x --full_resolution 1024 --channels 1 --save_name model --scale_factor 2 --testing_method model --model_name SSR_iso_mag --print True --device cuda:0 --parallel True --data_folder iso1024mag --mode 3D

#python3 test_SSR.py --output_file_name iso3D_4x --full_resolution 1024 --channels 1 --save_name trilinear --scale_factor 4 --testing_method trilinear --print True --device cuda:0 --data_folder iso1024mag --mode 3D 
#python3 test_SSR.py --output_file_name iso3D_4x --full_resolution 1024 --channels 1 --save_name model --scale_factor 4 --testing_method model --model_name SSR_iso_mag --print True --device cuda:0 --parallel True --data_folder iso1024mag --mode 3D

#python3 test_SSR.py --output_file_name iso3D_8x --full_resolution 1024 --channels 1 --save_name trilinear --scale_factor 8 --testing_method trilinear --print True --device cuda:0 --data_folder iso1024mag --mode 3D 
#python3 test_SSR.py --output_file_name iso3D_8x --full_resolution 1024 --channels 1 --save_name model --scale_factor 8 --testing_method model --model_name SSR_iso_mag --print True --device cuda:0 --parallel True --data_folder iso1024mag --mode 3D

#python3 test_SSR.py --output_file_name iso3D_16x --full_resolution 1024 --channels 1 --save_name trilinear --scale_factor 16 --testing_method trilinear --print True --device cuda:0 --data_folder iso1024mag --mode 3D 
#python3 test_SSR.py --output_file_name iso3D_16x --full_resolution 1024 --channels 1 --save_name model --scale_factor 16 --testing_method model --model_name SSR_iso_mag --print True --device cuda:0 --parallel True --data_folder iso1024mag --mode 3D

#################################################

#python3 -u test_SSR.py --output_file_name isoVF3D_2x --full_resolution 512 --channels 3 --save_name trilinear --scale_factor 2 --testing_method trilinear --print True --device cuda:0 --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_2x --full_resolution 512 --channels 3 --save_name model --scale_factor 2 --testing_method model --model_name SSR_isoVF --print True --device cuda:0 --parallel True --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true

#python3 -u test_SSR.py --output_file_name isoVF3D_4x --full_resolution 512 --channels 3 --save_name trilinear --scale_factor 4 --testing_method trilinear --print True --device cuda:0 --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_4x --full_resolution 512 --channels 3 --save_name model --scale_factor 4 --testing_method model --model_name SSR_isoVF --print True --device cuda:0 --parallel True --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true

#python3 -u test_SSR.py --output_file_name isoVF3D_8x --full_resolution 512 --channels 3 --save_name trilinear --scale_factor 8 --testing_method trilinear --print True --device cuda:0 --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_8x --full_resolution 512 --channels 3 --save_name model --scale_factor 8 --testing_method model --model_name SSR_isoVF --print True --device cuda:0 --parallel True --data_folder iso3DVF --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true

#################################################


#python3 -u test_SSR.py --output_file_name isoVF3D_2x --full_resolution 128 --channels 3 --save_name trilinear --scale_factor 2 --testing_method trilinear --print True --device cuda:0 --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_2x --full_resolution 128 --channels 3 --save_name model --scale_factor 2 --testing_method model --model_name SSR_isoVF_channelscaling --print True --device cuda:0 --parallel True --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true

#python3 -u test_SSR.py --output_file_name isoVF3D_4x --full_resolution 128 --channels 3 --save_name trilinear --scale_factor 4 --testing_method trilinear --print True --device cuda:0 --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_4x --full_resolution 128 --channels 3 --save_name model --scale_factor 4 --testing_method model --model_name SSR_isoVF_channelscaling --print True --device cuda:0 --parallel True --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true

#python3 -u test_SSR.py --output_file_name isoVF3D_8x --full_resolution 128 --channels 3 --save_name trilinear --scale_factor 8 --testing_method trilinear --print True --device cuda:0 --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
#python3 -u test_SSR.py --output_file_name isoVF3D_8x --full_resolution 128 --channels 3 --save_name model --scale_factor 8 --testing_method model --model_name SSR_isoVF_channelscaling --print True --device cuda:0 --parallel True --data_folder iso3D_VF_test --mode 3D --test_streamline True --fix_dim_order false --test_on_gpu true
