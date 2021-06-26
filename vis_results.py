import numpy as np
import matplotlib.pyplot as plt
import argparse
from test_SSR import load_obj
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Plume_16x_results",
    type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="Plume_16x",
    type=str,help='filename to visualize in output folder')
    parser.add_argument('--start_ts', default=21, type=int)
    parser.add_argument('--ts_skip', default=1, type=int)
    
    
    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    results_file = os.path.join(output_folder, args['output_file_name'])
    
    results = load_obj(results_file)
    save_folder = os.path.join(output_folder, args['save_folder'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    
    method_names = list(results.keys())
    metric_names = list(results[method_names[0]].keys())

    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 15}
    plt.rcParams.update(font)

    for metric in metric_names:
        fig = plt.figure()
        if(metric == "psnr"):
            y_label = "PSNR (dB)"
        elif(metric == "inner_psnr"):
            y_label = "Inner PSNR (dB)"
        elif(metric == "ssim"):
            y_label = "SSIM"
        else:
            y_label = metric

        for method in method_names:
            #if(method == "model"):
            #    continue
            x = np.arange(args['start_ts'], 
            args['start_ts'] + args['ts_skip']*len(results[method][metric]),
            args['ts_skip'])
            y = results[method][metric]
            if(len(y) > 0):
                print(method + " " + metric + " " + str(np.median(np.array(y)[1:])))
            plt.plot(x, y, label=method)
        plt.legend()
        plt.xlabel("Simulation timestep")
        plt.ylabel(y_label)
        #plt.title(args['output_file_name'] + " - " + metric)
        plt.savefig(os.path.join(save_folder, metric+".png"))
        #plt.show()
        plt.clf()
