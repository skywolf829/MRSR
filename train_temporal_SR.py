from temporal_models import *
from options import *
from utility_functions import *
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import os
import imageio
import argparse
import time
import datetime
from pytorch_memlab import LineProfiler, MemReporter, profile
from datasets import LocalTemporalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default=None,type=str,help='File to train on')
    parser.add_argument('--num_training_examples',default=None,type=int,help='Frames to use from training file')
    parser.add_argument('--save_folder',default=None, help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None, help='The name for the folder to save the model')
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels to use')
    parser.add_argument('--spatial_downscale_ratio',default=None,type=float,help='Ratio for spatial downscaling')
    parser.add_argument('--min_dimension_size',default=None,type=int,help='Minimum dimension size')
    parser.add_argument('--scaling_mode',default=None,type=str,help='Scaling mode, learned, magnitude, channel, or none')

    
    parser.add_argument('--num_lstm_layers',default=None,type=int,help='num lstm layers')
    parser.add_argument('--training_seq_length',default=None,type=int,help='length of sequence to train LSTM with')
    parser.add_argument('--temporal_direction',default=None,type=str,help='forward or backward')
    parser.add_argument('--temporal_model',type=str,default=None, help='Model to use for temporal inteprolation')

    parser.add_argument('--num_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--base_num_kernels',default=None,type=int, help='Num conv kernels in lowest layer')
    parser.add_argument('--pre_padding',default=None,type=str2bool, help='Padding before entering network')
    parser.add_argument('--kernel_size',default=None, type=int,help='Conv kernel size')
    parser.add_argument('--stride',default=None, type=int,help='Conv stride length')
    
    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--device',type=str,default=None, help='Device to use')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--ranking',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--save_generators',default=None, type=str2bool,help='Whether or not to save generators')
    parser.add_argument('--save_discriminators',default=None, type=str2bool,help='Whether or not to save discriminators')

    parser.add_argument('--alpha_1',default=None, type=float,help='Reconstruction loss coefficient')
    parser.add_argument('--alpha_2',default=None, type=float,help='Adversarial loss coefficient')
    parser.add_argument('--alpha_3',default=None, type=float,help='Soft physical constraint loss coefficient')
    parser.add_argument('--alpha_4',default=None, type=float,help='mag and angle loss coeff')
    parser.add_argument('--alpha_5',default=None, type=float,help='gradient loss coeff')
    parser.add_argument('--alpha_6',default=None, type=float,help='streamline loss coeff')
    parser.add_argument('--streamline_res',default=None, type=int,help='num seeds per dim')
    parser.add_argument('--streamline_length',default=None, type=int,help='timesteps to do streamlines')
    parser.add_argument('--adaptive_streamlines',default=None, type=str2bool,help='Adaptive particle sampling for streamlines')
    parser.add_argument('--periodic',default=None, type=str2bool,help='is data periodic')
    parser.add_argument('--generator_steps',default=None, type=int,help='Number of generator steps to take')
    parser.add_argument('--discriminator_steps',default=None, type=int,help='Number of discriminator steps to take')
    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--minibatch',default=None, type=int,help='Size of minibatch to train on')
    parser.add_argument('--num_workers',default=None, type=int,help='Number of workers for dataset loader')
    parser.add_argument('--g_lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--d_lr',default=None, type=float,help='Learning rate for the discriminator')    
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')
    parser.add_argument('--gamma',default=None, type=float,help='')
    parser.add_argument('--regularization',default=None, type=str,help='')
    parser.add_argument('--physical_constraints',default=None,type=str,help='none, soft, or hard')
    parser.add_argument('--patch_size',default=None, type=int,help='Patch size for inference')
    parser.add_argument('--training_patch_size',default=None, type=int,help='Patch size for training')
    parser.add_argument('--upsample_mode',default=None, type=str,help='Method for upsampling')
    parser.add_argument('--zero_noise',default=None, type=str2bool,help='Whether or not to use zero noise during upscaling')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')


    args = vars(parser.parse_args())

    FlowSTSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(FlowSTSR_folder_path, "InputData")
    output_folder = os.path.join(FlowSTSR_folder_path, "Output")
    save_folder = os.path.join(FlowSTSR_folder_path, "SavedModels")


    if(args['load_from'] is None):
        
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
                
        dataset = LocalTemporalDataset(opt)
        if(opt['temporal_model'] == "TSRTVD"):
            generator = Temporal_Generator(opt)
        elif(opt['temporal_model'] == "UNet"):
            generator = Temporal_Generator_UNET(opt)  
        generator.apply(weights_init)
        discriminator = Temporal_Discriminator(opt)
        discriminator.apply(weights_init)

    else:        
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        generator = Temporal_Generator(opt)
        discriminator = Temporal_Discriminator(opt)
        generator, discriminator = load_models(generator,discriminator,opt,args["device"])
        dataset = LocalTemporalDataset(opt)

    now = datetime.datetime.now()
    start_time = time.time()
    print_to_log_and_console("Started training at " + str(now), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    opt["num_training_examples"] = len(os.listdir(opt["data_folder"]))

   
    start_time_scale_n = time.time()

    print_to_log_and_console(str(datetime.datetime.now()) + " - Beginning training.",
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    #with profiler.profile(profile_memory=True, use_cuda=True, record_shapes=True) as prof:
    if(opt['train_distributed']):
        os.environ['MASTER_ADDR'] = '127.0.0.1'              
        os.environ['MASTER_PORT'] = '29500' 
        mp.spawn(train_temporal_network,
            args=(generator, discriminator,opt,dataset),
            nprocs=opt['gpus_per_node'],
            join=True)
    else:
        generator = train_temporal_network(opt['device'], generator, discriminator, opt, dataset)
    
    time_passed = (time.time() - start_time) / 60
    print_to_log_and_console("%s - Finished training  in %f minutes" % (str(datetime.datetime.now()), time_passed),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 



