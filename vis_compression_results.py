import numpy as np
import matplotlib.pyplot as plt
import argparse
from utility_functions import load_obj
import os
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="mag3D_compression",
    type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="results.pkl",
    type=str,help='filename to visualize in output folder')    
    
    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['save_folder'])
    results_file = os.path.join(save_folder, args['output_file_name'])
    
    make_all = False
    #full_file_size = 524209
    #full_file_size = 4096
    full_file_size = 4194306
    results = load_obj(results_file)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    '''
    results.pop('sz', None)
    results.pop("NN_SZ", None)
    results.pop("NN_bilinearheuristic_mixedLOD_octree_SZ")
    results["Super resolution + SZ"] = copy.deepcopy(results["NN_mixedLODoctree_SZ"])
    results.pop("NN_mixedLODoctree_SZ")

    for m in results.keys():
        ks = list(results[m].keys())
        for k in ks:
            if("innner" in k):
                print(m)
                k_orig = k.split("innner")[0] + "inner" + k.split("innner")[1]
                results[m][k_orig] = copy.deepcopy(results[m][k])
                results[m].pop(k,None)
    # results go compression_method -> metric -> list
    '''

    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 18}
    plt.rcParams.update(font)
    for method in ["NN_trilinearheuristic_SR_octree_SZ", "SZ"]:#, "NN_SZ"]:
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
    for i in range(len(results['NN_trilinearheuristic_SR_octree_SZ']['psnrs'])):
        print("%i target %f: PSNR %0.02f SSIM %0.04f CR %0.01f" % \
            (i, 
            results['NN_trilinearheuristic_SR_octree_SZ']['psnrs'][i],
            results['NN_trilinearheuristic_SR_octree_SZ']['rec_psnr'][i], 
            results['NN_trilinearheuristic_SR_octree_SZ']['rec_ssim'][i],
            results['NN_trilinearheuristic_SR_octree_SZ']['compression_ratio'][i]))
    
    if(not make_all):
        

        fig = plt.figure()    

        x = np.array(results["NN_trilinearheuristic_SR_octree_SZ"]['TKE_error'][:])
        y = np.array(results["NN_trilinearheuristic_SR_octree_SZ"]["compression_ratio"])
        for i in range(0):
            y = np.delete(y, x.argmin())
            x = np.delete(x, x.argmin())
        for i in range(14):
            y = np.delete(y, x.argmax())
            x = np.delete(x, x.argmax())
        ordering = np.argsort(x)        
        x = x[ordering]        
        y = y[ordering]
        plt.plot(y, x, label="Ours")

        '''
        x = np.array(results["NN_SZ"]['rec_ssim'][:])
        y = np.array(results["NN_SZ"]["compression_ratio"])
        x[x>50] = 50
        ordering = np.argsort(x)        
        x = x[ordering]        
        y = y[ordering]
        plt.step(y, x, label="Ours without dynamic downscaling", where='post', linestyle='--')
        plt.plot(y, x, 'C2o', alpha=0.5, color = "orange")
        '''
        x = np.array(results["SZ"]['TKE_error'])
        y = np.array(results["SZ"]["compression_ratio"])
        for i in range(2):
            y = np.delete(y, x.argmin())
            x = np.delete(x, x.argmin())
        for i in range(0):
            y = np.delete(y, x.argmax())
            x = np.delete(x, x.argmax())
        ordering = np.argsort(x)
        x = x[ordering]        
        y = y[ordering]
        plt.plot(y, x, label="SZ alone")
        
            
        plt.legend()
        plt.ylabel("(De)compressed PSNR (dB)")
        plt.xlabel("Compression ratio")
        plt.title("(De)compressed PSNR (dB) over Compression Ratios")
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
                