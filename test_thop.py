# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Jiaqing Zhang & Kai Jiang
# ------------------------------------------------------------------------------


import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/cityscapes/seg_hrnet_AWCA_PSNL_z_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', #修改
                        #required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--continue_training",
                        help="continue_training", 
                        type=str, 
                        default='1',
                        choices=['0', '1'])

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )        

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    ################ test the model size ################
    import thop
    image = torch.randn((1,10,200,200))
    flops, params = thop.profile(model, inputs=(image,), verbose=True)
    print('Params: %.4fM'%(params/1e6))
    print('FLOPs: %.2fG'%(flops/1e9))


if __name__ == '__main__':
    main()
