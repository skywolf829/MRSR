#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/FlowSTSR
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_2x --save_name model --SR_factor 2 --testing_method model --print True --device cuda
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_2x --save_name trilinear --SR_factor 2 --testing_method trilinear --print True --device cuda
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_4x --save_name model --SR_factor 4 --testing_method model --print True --device cuda
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_4x --save_name trilinear --SR_factor 4 --testing_method trilinear --print True --device cuda
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_8x --save_name model --SR_factor 8 --testing_method model --print True --device cuda
python3 test_SSR.py --test_energy_spectra False --output_file_name SSR_8x --save_name trilinear --SR_factor 8 --testing_method trilinear --print True --device cuda