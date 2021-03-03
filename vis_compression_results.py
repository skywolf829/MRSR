import numpy as np
import matplotlib.pyplot as plt
import argparse
from test_compression import load_obj
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="mag2D_4010",
    type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="results.pkl",
    type=str,help='filename to visualize in output folder')    
    
    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['save_folder'])
    results_file = os.path.join(save_folder, args['output_file_name'])
    
    results = load_obj(results_file)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    compression_names = list(results.keys())
    psnrs = list(results[compression_names[0]].keys())

    for metric in psnrs:
        fig = plt.figure()

        for method in method_names:
            x = np.arange(args['start_ts'], 
            args['start_ts'] + args['ts_skip']*len(results[method][metric]),
            args['ts_skip'])
            y = results[method][metric]
            plt.plot(x, y, label=method)
        plt.legend()
        plt.xlabel("Simulation timestep")
        plt.ylabel(metric)
        plt.title(args['output_file_name'] + " - " + metric)
        plt.savefig(os.path.join(save_folder, metric+".png"))
        plt.clf()
