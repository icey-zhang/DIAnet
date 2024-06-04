# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Jiaqing Zhang & Kai Jiang
# ------------------------------------------------------------------------------


import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
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
######## ACW_loss ########
from core.criterion import CrossEntropy, OhemCrossEntropy, ACWloss
######## ACW_loss ########
from core.function import train, validate, validate_patch
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

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
                        default='0',
                        choices=['0', '1'])
    ######## Attention ########
    parser.add_argument("--HSN_position", type=str, default='0',
                        choices=['0', '1', '2', '3', '1+2+3', '2+3'])

    parser.add_argument("--PSNL_position", type=str, default='0',
                        choices=['0', '1', '2', '3', '1+2+3', '2+3'])

    parser.add_argument("--Attention_order", type=str, default='0',
                        choices=['0', 'H', 'P', 'HP', 'PH', 'H/P'])
    ######## Attention ########

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

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

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR,
                        )

    print('train_dataset load name and path')

    train_sampler = get_sampler(train_dataset)
    print(train_sampler)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)
    

    print('train_dataset load success')


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1,
                        )

    print('test_dataset loade name and path')

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)
    
    print('test_dataset load success')

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    ######## ACW_loss ########
    elif config.LOSS.USE_ACW:
        criterion = ACWloss(ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights)
        print('use ACW loss')
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)
        print('use Cross entropy')
    ######## ACW_loss ########
    

    print('criterion load success')

    model = FullModel(model, criterion)
    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        print('distributed training')
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        print('non-distributed training')

    ######## Attention ########
    if args.Attention_order == '0':
        print('Attention Module: None')
    elif args.Attention_order == 'H':
        print('Attention Module: HSN')
    elif args.Attention_order == 'P':
        print('Attention Module: PSNL')
    elif args.Attention_order == 'HP':
        print('Attention Module: HSN+PSNL')
    elif args.Attention_order == 'PH':
        print('Attention Module: PSNL+HSN')
    elif args.Attention_order == 'H/P':
        print('Attention Module: HSN and PSNL(parallel)')

    if args.HSN_position != '0':
        print('HSN position: %s' % args.HSN_position)
    if args.PSNL_position != '0':
        print('PSNL position: %s' % args.PSNL_position)
    ######## Attention ########
    

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int_(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    best_FwIoU = 0.0
    FwIoU = 0.0
    last_epoch = 0
    #### continue ####
    if args.continue_training == '1':
    #if False:
    #### continue ####
        model_state_file = os.path.join('xxxx') #修改路径继续训练
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cuda:0')
            try:
                best_FwIoU = checkpoint['best_FwIoU']
            except:
                print("checkpoint未记录best_FwIoU")
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    
    for epoch in range(last_epoch, end_epoch):
        print('1')

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
            print('2')

        # valid_loss, mean_IoU, IoU_array = validate(config, 
        #             testloader, model, writer_dict)
        """"""
        train(config, epoch, config.TRAIN.END_EPOCH, 
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)
        print('3')
        print('\ntraining DIANet\n')
        print (time.strftime('%H:%M:%S',time.localtime(time.time())))
        
        valid_loss, mean_IoU, IoU_array, FwIoU = validate_patch(config, 
                    testloader, model, writer_dict)
        print('4')
        
        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_FwIoU': best_FwIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
            if FwIoU > best_FwIoU:
                best_FwIoU = FwIoU
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, FwIoU: {: 4.4f}, best_FwIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, FwIoU, best_FwIoU)
            logging.info(msg)
            logging.info(IoU_array)

    if args.local_rank <= 0:

        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int_((end-start)/3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
