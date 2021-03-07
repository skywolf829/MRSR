#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR

python3 test_SSR.py --output_file_name iso2D_LW_2x --full_resolution 1024 --channels 1 --save_name model --scale_factor 2 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_2x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 2 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_2x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 2 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_LW_4x --full_resolution 1024 --channels 1 --save_name model --scale_factor 4 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_4x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 4 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_4x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 4 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_LW_8x --full_resolution 1024 --channels 1 --save_name model --scale_factor 8 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_8x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 8 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_8x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 8 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_LW_16x --full_resolution 1024 --channels 1 --save_name model --scale_factor 16 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_16x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 16 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_16x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 16 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_LW_32x --full_resolution 1024 --channels 1 --save_name model --scale_factor 32 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_32x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 32 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_32x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 32 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 

python3 test_SSR.py --output_file_name iso2D_LW_64x --full_resolution 1024 --channels 1 --save_name model --scale_factor 64 --testing_method model --model_name SSR_isomag2D_LW --print True --device cuda:0 --parallel False --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_64x --full_resolution 1024 --channels 1 --save_name bilinear --scale_factor 64 --testing_method bilinear --print True --device cuda:0 --data_folder isomag2D --mode 2D 
python3 test_SSR.py --output_file_name iso2D_LW_64x --full_resolution 1024 --channels 1 --save_name bicubic --scale_factor 64 --testing_method bicubic --print True --device cuda:0 --data_folder isomag2D --mode 2D 


#python3 test_SSR.py --output_file_name mixing3D_2x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 2 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_2x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 2 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_4x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 4 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_4x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 4 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_8x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 8 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_8x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 8 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_16x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 16 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_16x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 16 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_32x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 32 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_32x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 32 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 

#python3 test_SSR.py --output_file_name mixing3D_64x --full_resolution 512 --channels 1 --save_name mixing3D_model --scale_factor 64 --testing_method model --model_name SSR_mixing_p --print True --device cuda:0 --parallel False --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name mixing3D_64x --full_resolution 512 --channels 1 --save_name mixing3D_bilinear --scale_factor 64 --testing_method trilinear --print True --device cuda:0 --data_folder mix_p --mode 3D 