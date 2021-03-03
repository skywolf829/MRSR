#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
#python3 test_SSR.py --output_file_name mixing_p_3D_2x --full_resolution 512 --channels 1 --save_name mixing3D --scale_factor 2 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel True --data_folder iso1024mag --mode 3D 
#python3 test_SSR.py --output_file_name mixing_p_3D_2x --full_resolution 512 --channels 1 --save_name mixing3D_trilinear --scale_factor 2 --testing_method trilinear --model_name SSR_mixing_p --print True --device cuda:0 --data_folder iso1024mag --mode 3D 
python3 test_SSR.py --output_file_name isomag2D_2x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 2 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_2x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 2 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name isomag2D_4x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 4 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_4x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 4 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 


python3 test_SSR.py --output_file_name isomag2D_8x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 8 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_8x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 8 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 


python3 test_SSR.py --output_file_name isomag2D_16x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 16 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_16x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 16 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 


python3 test_SSR.py --output_file_name isomag2D_32x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 32 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_32x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 32 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 


python3 test_SSR.py --output_file_name isomag2D_64x --full_resolution 1024 --channels 1 --save_name isomag2D_model --scale_factor 64 --testing_method model --model_name SSR_isomag2D --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name isomag2D_64x --full_resolution 1024 --channels 1 --save_name isomag2D_bilinear --scale_factor 64 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 