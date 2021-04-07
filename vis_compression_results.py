import numpy as np
import matplotlib.pyplot as plt
import argparse
from utility_functions import load_obj, save_obj
import os
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="mixing3D_compression_test",
    type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="results.pkl",
    type=str,help='filename to visualize in output folder')    
    
    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['save_folder'])
    results_file = os.path.join(save_folder, args['output_file_name'])
    
    make_all = False
    full_file_size = 524209
    #full_file_size = 4096
    #full_file_size = 4194306
    results = load_obj(results_file)
    save_obj(results, results_file)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
   
    tester = "NN_SZ"

    print(results.keys())
 
    for metric in results["SZ"].keys():
        results["SZ"][metric] = np.delete(results["SZ"][metric], [168])
        #results[tester][metric].pop(0)
    #for metric in results[tester].keys():
       #results[tester][metric] = np.delete(results[tester][metric], [np.arange(0,51)])
        #results[tester][metric].pop(0)

    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 18}
    plt.rcParams.update(font)
    for method in [tester, 'SZ']:
        file_size = results[method]['file_size']
        compression_ratios = full_file_size / np.array(file_size)
        results[method]['compression_ratio'] = compression_ratios

    print(results['SZ'].keys())
    print("SZ")
    for i in range(len(results['SZ']['psnrs'])):
        print("%i target %f: PSNR %0.02f SSIM %0.04f CR %0.01f" % \
            (i, 
            results['SZ']['psnrs'][i],
            results['SZ']['rec_psnr'][i], 
            results['SZ']['rec_ssim'][i],
            results['SZ']['compression_ratio'][i]))

    print("Ours")
    for i in range(len(results[tester]['psnrs'])):
        print("%i target %f: PSNR %0.02f SSIM %0.04f CR %0.01f CT %0.01f" % \
            (i, 
            results[tester]['psnrs'][i],
            results[tester]['rec_psnr'][i], 
            results[tester]['rec_ssim'][i],
            results[tester]['compression_ratio'][i],
            results[tester]['compression_time'][i])
            )

    if(not make_all):
        

        fig = plt.figure()    

        x = np.array(results[tester]['rec_psnr'][:])
        y = np.array(results[tester]["compression_ratio"])
        for i in range(1):
            y = np.delete(y, x.argmin())
            x = np.delete(x, x.argmin())
        for i in range(0):
            y = np.delete(y, x.argmax())
            x = np.delete(x, x.argmax())
        ordering = np.argsort(x)        
        x = x[ordering]        
        y = y[ordering]
        plt.plot(y, x, label="Ours")

        
        #x = np.array([53.96, 50.94, 40.83, 37.77])
        #x = np.array([1.0, 0.9864, 0.8833, 0.8063])
        #y = np.array([1, 87.5, 691.0, 940.9])

        #x = np.array([51.69, 45.04, 38.08, 36.27])
        #x = np.array([1.0, .9991, 0.9934, 0.9892])
        #y = np.array([1, 442.6, 3543.8, 4846.7])

        ordering = np.argsort(x)        
        x = x[ordering]        
        y = y[ordering]
        plt.step(y, x, label="Ours without dynamic downscaling", where='post', linestyle='--')
        plt.plot(y, x, 'C2o', alpha=0.5, color = "orange")
        
        
        x = np.array(results["SZ"]['rec_psnr'])
        y = np.array(results["SZ"]["compression_ratio"])
        for i in range(0):
            y = np.delete(y, x.argmin())
            x = np.delete(x, x.argmin())
        for i in range(0):
            y = np.delete(y, x.argmax())
            x = np.delete(x, x.argmax())
        ordering = np.argsort(x)
        x = x[ordering]        
        y = y[ordering]
        plt.plot(y, x, label="SZ alone")
        
            
        #plt.legend()
        plt.ylabel("Reconstructed data PSNR (dB)")
        plt.xlabel("Compression ratio")
        plt.title("Reconstructed data PSNR over Compression Ratios")
        plt.tight_layout()
        #plt.savefig(os.path.join(save_folder, metric+"_psnr.png"))
        plt.show()
        plt.clf()
    else:
        for m in results.keys():
            print(m)
            for k in results[m].keys():
                print("    " + k)
                if(k == "rec_psnr" or k == "rec_ssim"\
                    or k == "file_size" or k == "psnrs" \
                        or k == "TKE_error" or k == "compression_time"):
                    print("        " + str(results[m][k]))

        compression_method_names = list(results.keys())
        metrics = ['file_size', 'compression_time', 'num_nodes', 'rec_psnr',
        'rec_mre', 'rec_pwmre', 'rec_inner_mre', 'rec_inner_pwmre', "TKE_error"]
        

        for metric in metrics:
            fig = plt.figure()
            vals = []
            if(metric == "file_size" and full_file_size is not None):
                metrics.append("compression_ratio")
                for method in compression_method_names:
                    file_size = results[method][metric]
                    compression_ratios = full_file_size / np.array(file_size)
                    results[method]['compression_ratio'] = compression_ratios
            for method in compression_method_names:
                if('rec_psnr' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0):
                    ordering = np.argsort(np.array(results[method]['rec_psnr'][:]))
                    x = np.array(results[method]['rec_psnr'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        x[x>55] = 55
                        plt.plot(x, y, label=method, drawstyle='steps')
                    elif "NN_trilinearheuristic_mixedLOD_octree_SZ" == method:
                        plt.plot(x, y, label=method)
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("(De)compressed PSNR")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " (de)compressed psnr vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_recpsnr.png"))
            plt.clf()
        
        
        for metric in metrics:
            fig = plt.figure()
            vals = []
            for method in compression_method_names:
                if('psnrs' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0 \
                        and method != "zfp"):
                    ordering = np.argsort(np.array(results[method]['psnrs'][:]))
                    x = np.array(results[method]['psnrs'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("Target PSNR")
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
                    ordering = np.argsort(np.array(results[method]['rec_ssim'][:]))
                    x = np.array(results[method]['rec_ssim'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    elif "NN_trilinearheuristic_mixedLOD_octree_SZ" == method:
                        plt.plot(x[:], y[:], label=method)
                    else:
                        plt.plot(x[:], y[:], label=method)
            plt.legend()
            plt.xlabel("(De)compressed SSIM")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " ssim vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_ssim.png"))
            plt.clf()
        '''
        for metric in metrics:
            fig = plt.figure()
            vals = []
            for method in compression_method_names:
                if('rec_mre' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0):
                    ordering = np.argsort(np.array(results[method]['rec_mre'][:]))
                    x = np.array(results[method]['rec_mre'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("(De)compressed maximum relative error (global)")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " MRE vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_mre.png"))
            plt.clf()
        
        for metric in metrics:
            fig = plt.figure()
            vals = []
            for method in compression_method_names:
                if('rec_inner_mre' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0):
                    ordering = np.argsort(np.array(results[method]['rec_inner_mre'][:]))
                    x = np.array(results[method]['rec_inner_mre'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("(De)compressed inner maximum relative error (global)")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " inner MRE vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_innermre.png"))
            plt.clf()
        
        for metric in metrics:
            fig = plt.figure()
            vals = []
            for method in compression_method_names:
                if('rec_pwmre' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0):
                    ordering = np.argsort(np.array(results[method]['rec_pwmre'][:]))
                    x = np.array(results[method]['rec_pwmre'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("(De)compressed pointwise maximum relative error (local)")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " PWMRE vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_pwmre.png"))
            plt.clf()
        
        for metric in metrics:
            fig = plt.figure()
            vals = []
            for method in compression_method_names:
                if('rec_innner_pwmre' in results[method].keys() and \
                    metric in results[method].keys() and len(results[method][metric]) > 0):
                    ordering = np.argsort(np.array(results[method]['rec_innner_pwmre'][:]))
                    x = np.array(results[method]['rec_innner_pwmre'])[ordering]
                    y = np.array(results[method][metric])[ordering]
                    if "NN_SZ" == method:
                        plt.plot(x, y, label=method, drawstyle='steps')
                    else:
                        plt.plot(x, y, label=method)
            plt.legend()
            plt.xlabel("(De)compressed inner pointwise maximum relative error (global)")
            plt.ylabel(metric)
            plt.title(args['output_file_name'] + " inner PWMRE vs - " + metric)
            plt.savefig(os.path.join(save_folder, metric+"_innerpwmre.png"))
            plt.clf()
        '''
        if("TKE_error" in metrics):
            for metric in metrics:
                fig = plt.figure()
                vals = []
                for method in compression_method_names:
                    if('TKE_error' in results[method].keys() and \
                        metric in results[method].keys() and len(results[method][metric]) > 0):
                        ordering = np.argsort(np.array(results[method]['TKE_error'][:]))
                        x = np.array(results[method]['TKE_error'])[ordering]
                        y = np.array(results[method][metric])[ordering]
                        if "NN_SZ" == method:
                            plt.plot(x, y, label=method, drawstyle='steps')
                        else:
                            plt.plot(x, y, label=method)
                plt.legend()
                plt.xlabel("(De)compressed total kinetic energy error")
                plt.ylabel(metric)
                plt.title(args['output_file_name'] + " TKE error vs - " + metric)
                plt.savefig(os.path.join(save_folder, metric+"_TKEerror.png"))
                plt.clf()
        if("compression_ratio" in metrics):
            for metric in metrics:
                fig = plt.figure()
                vals = []
                print(metric)
                for method in compression_method_names:
                    if('compression_ratio' in results[method].keys() and \
                        metric in results[method].keys() and len(results[method][metric]) > 0):
                        ordering = np.argsort(np.array(results[method]['compression_ratio'][:]))
                        x = np.array(results[method]['compression_ratio'])[ordering]
                        y = np.array(results[method][metric])[ordering]
                        if "NN_SZ" == method:
                            plt.plot(x, y, label=method, drawstyle='steps')
                        else:
                            plt.plot(x, y, label=method)
                plt.legend()
                plt.xlabel("Compression ratio")
                plt.ylabel(metric)
                plt.title(args['output_file_name'] + " Compression Ratio vs - " + metric)
                plt.savefig(os.path.join(save_folder, metric+"_compressionratio.png"))
                plt.clf()
                
print("Compression time median: " + str(np.median(np.array(results[tester]["compression_time"]))))
#del results['NN_trilinearheuristic_SR_octree_SZ']
#save_obj(results, results_file)