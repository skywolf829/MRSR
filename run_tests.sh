#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 test_SSR.py --output_file_name mixing_p_3D_2x --save_name mixing3D --scale_factor 2 --testing_method model --print True --device cuda:0 --parallel True --data_folder mix_p --mode 3D 
#python3 test_SSR.py --output_file_name isomag3D_4x --save_name model_nostreamlines --scale_factor 4 --testing_method model --print True --device cuda:0 --parallel True
#python3 test_SSR.py --output_file_name isomag3D_8x --save_name model_nostreamlines --scale_factor 8 --testing_method model --print True --device cuda:0 --parallel True
