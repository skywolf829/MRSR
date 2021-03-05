import numpy as np
import matplotlib.pyplot as plt
import argparse
from test_SSR import load_obj
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="mag2D_4010_psnr_compression",
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
    print(results)

    # results go compression_method -> metric -> list

    compression_method_names = list(results.keys())
    metrics = ['file_size', 'compression_time', 'num_nodes', 'rec_psnr']

    for metric in metrics:
        fig = plt.figure()
        vals = []
        for method in compression_method_names:
            if('rec_psnr' in results[method].keys() and \
                metric in results[method].keys() and len(results[method][metric]) > 0):
                x = np.array(results[method]['rec_psnr'][2:7])
                y = results[method][metric][2:7]
                plt.plot(x, y, label=method)
        plt.legend()
        plt.xlabel("(De)compressed PSNR")
        plt.ylabel(metric)
        plt.title(args['output_file_name'] + " psnr vs - " + metric)
        plt.savefig(os.path.join(save_folder, metric+"_psnr.png"))
        plt.clf()
    for metric in metrics:
        fig = plt.figure()
        vals = []
        for method in compression_method_names:
            if('rec_ssim' in results[method].keys() and \
                metric in results[method].keys() and len(results[method][metric]) > 0):
                x = np.array(results[method]['rec_ssim'][2:7])
                y = results[method][metric][2:7]
                plt.plot(x, y, label=method)
        plt.legend()
        plt.xlabel("(De)compressed SSIM")
        plt.ylabel(metric)
        plt.title(args['output_file_name'] + " ssim vs - " + metric)
        plt.savefig(os.path.join(save_folder, metric+"_ssim.png"))
        plt.clf()