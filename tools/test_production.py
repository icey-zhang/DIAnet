import torch

import os

import argparse
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import scipy.io as sio
import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy, ACWloss
from core.function import validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive_for_awca_psnl#, validate2_patch_ensemble_for_nc_1000_1400
from utils.utils import create_logger
os.environ['CUDA_VISIBLE_DEVICES']='0'
from IO.fy_testdatasave import scatter_Asian
from IO.fy_testdatasave import save_hdf
from datasets.fy_testdata import correct_GF
import torch.nn.functional as F
#######################################################################
############################### 地址修改 ##################################################

root_address = './'

root_HDF_img_address = './datasets/HDF'


############################### 可视化绘图 ################################################
label_mapping = {0: 0, 1: 1, 2: 2, 3: 3,4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def save_pred(preds, sv_path, name):
    palette = get_palette(256)
    palette[27:30] = palette[33:36]
    palette[30:33] = [255,255,255]
    try:
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        #label = np.asarray(label.cpu(), dtype=np.uint8).squeeze()
    except:
        preds = preds.transpose(2,0,1)
        preds = preds.reshape(1,preds.shape[0],preds.shape[1],preds.shape[2])
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        #label = np.asarray(label, dtype=np.uint8).squeeze()
        #print("输入作为numpy处理")
    for i in range(preds.shape[0]):
        pred = convert_label(preds[i], inverse=True)
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, ''.join(name)+'.png'))
    """
    label = convert_label(label, inverse=True)
    save_img = Image.fromarray(label)
    save_img.putpalette(palette)
    save_img.save(os.path.join(sv_path, ''.join(name)+'_label.png'))
    """
def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label
def save_pred_channel_1_as_mat(preds, sv_path, name):
    save_fn = sv_path+'/'+name[0][:-4]+'.mat'
    save_array = preds.cpu().numpy().squeeze()
    save_array = save_array.transpose(1,2,0)
    sio.savemat(save_fn, {'data': save_array})
##########################################################################################
class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=None,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        # logx.msg("Using Cross Entropy Loss")
        print("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets, do_rmi=None):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
##########################################################################################

def parse_args_for_HR_AWCA_PSNL(model_name):
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=os.path.join(root_address, 'experiments/cityscapes/seg_'+model_name+'_w48_train_100x100_sgd_lr1e-2_wd5e-4_bs_8_epoch100.yaml'),
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
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

def test_model_on_gpu_for_hr_based(model_name,patch_mode,data_mode):
    """
    测试以 nvida 为 backbone 的网络
    """
    args = parse_args_for_HR_AWCA_PSNL(model_name)


    if args.seed > 0:
        import random   
        #print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

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
    if model_name[:9] != 'hrnet_PSP':
        model = eval('models.'+config.MODEL.NAME +
                    '.get_seg_model')(config)

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
    train_dataset = eval('datasets.'+config.DATASET.DATASET+'_for_full_nc_without_gtmap_norm2')(
                        root=config.DATASET.ROOT,  ## address 改
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
                        model_name = model_name,
                        data_mode = data_mode,
                        root_HDF_img = root_HDF_img_address)

    #print('train_dataset load name and path')

    train_sampler = get_sampler(train_dataset)
    #print(train_sampler)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET+'_for_full_nc_without_gtmap_norm2')(
                        root=config.DATASET.ROOT,  ## address 改
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=1,
                        crop_size=test_size,
                        downsample_rate=1,
                        model_name = model_name,
                        data_mode = data_mode,
                        root_HDF_img = root_HDF_img_address)


    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    

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
    
    # PSP_model 在此处 build model
    if model_name[:9] == 'hrnet_PSP':
        model = eval('models.'+config.MODEL.NAME +
                    '.get_seg_model')(config, criterion)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        #print('distributed training')
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        #model = model.cuda()
        print('testing on gpu, without nn.DataParallel')

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
            #print(nbb_keys)
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
    last_epoch = 0
    if True:

        model_state_file = root_address + config.TEST.MODEL_FILE
        print(model_state_file)
        assert os.path.isfile(model_state_file)
        if os.path.isfile(model_state_file):
            
            checkpoint = torch.load(model_state_file)
            best_mIoU = 0
            last_epoch = 0
            dct = checkpoint
            if model_name[:9] == 'hrnet_PSP':
                # 读取 PSP_model 的权重
                model.module.load_state_dict({k: v for k, v in checkpoint.items()})
            elif model_name[:-2]=='hrnet_ACW_sub_model':
                # 读取其它模型的权重
                model.module.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            else:
                model.module.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})
            print('loaded trained model!!!')
            #model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})
            """"""
            #######################################################################
            
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    
    for epoch in range(last_epoch, 1):
        print(model_name)
        print(epoch)
        if model_name[:-2] == 'hrnet_AWCA_PSNL':
            validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive_for_awca_psnl(config, test_dataset, testloader, model, writer_dict, model_name,patch_mode=patch_mode,root_address=root_address, sub_batch_size=3)

############################### ensemble model #######################################################################################
def ensemble_model():
    #####计时开始#####
    start = time.time()
    ###加
    #  HDF-->mat并保存
    data_mode = 'HDF'#'mat'
    patch_mode = 'on'           # 'on':进行裁剪后输入网络，通过网络后拼接，（800，1000）；‘off’:不进行裁剪，（1000，1400）
    fy_path = root_HDF_img_address
    save_path = root_HDF_img_address[:-3]+'HDF_mat'
    ###加
    size_w = 1000
    size_h = 1400
    """"""

    for model_name in ['hrnet_AWCA_PSNL_z']:
        test_model_on_gpu_for_hr_based(model_name, patch_mode, data_mode)  # 加

    ############ Set up metrics ##########
    start_time = time.time()
    idx = 0
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_datatset_ = eval('datasets.'+config.DATASET.DATASET+'_for_full_nc_without_gtmap_norm2')(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.TEST_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=False,
                    flip=False,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TEST.BASE_SIZE,
                    crop_size=test_size,
                    downsample_rate=1, 
                    model_name = model_name
                    data_mode = data_mode,
                    root_HDF_img = root_HDF_img_address)  
    for file_info in test_datatset_.files:
        file_name = file_info['name']
        ensemble_data = np.zeros([size_w, size_h, 11])
        i=0
        factor = {'hrnet_AWCA_PSNL_z':1}
                ###加###
        weight_class = {'hrnet_AWCA_PSNL_z':[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],}        
        ###加###
        for model_name in [ 'hrnet_AWCA_PSNL_z']:
            # 读取保存的单个模型的结果          
            load_fn = root_address + '/tools'+'//'+ model_name+'_nc_results2'+'/prob'+'/'+file_name+'.mat'
            load_data = sio.loadmat(load_fn)
            load_matrix = load_data['data']
            #print(load_matrix.shape)
            if model_name == 'Dplv3p_psp' or model_name == 'hr_plv3p_v1':      # [11,:,:]   model_name == 'gated' or
                ###加###
                weight_class_i = np.array(weight_class[model_name]).reshape(1,1,11)
                ensemble_data += load_matrix.squeeze().transpose(1,2,0) * weight_class_i * factor[model_name]    # factor保存了集成的权重  [:,:,11]
                ###加###
            else:
                ###加###
                weight_class_i = np.array(weight_class[model_name]).reshape(1,1,11)
                ensemble_data += load_matrix * weight_class_i * factor[model_name]   # [:,:,11]
                ###加###
            #ensemble_data[:,:,model_class_num] = load_matrix[:,:,model_class_num]
            i += 1
        # ensemble_result = np.argmax(ensemble_data,axis=2)
        
        # #####计时结束#####
        end = time.time()
        print('all model:%d'%(end - start))
        ################################
        ensemble_result = np.argmax(ensemble_data,axis=2)
        ensemble_path = root_address + '/RESULT/'

        if not os.path.exists(ensemble_path):
            os.mkdir(ensemble_path)
            os.mkdir(ensemble_path+'/CloudImage/')
            os.mkdir(ensemble_path+'/CloudProduct/')

        save_pred(preds=ensemble_data, sv_path=ensemble_path, name=file_name)                                 
        scatter_Asian(mask = ensemble_result,save_path_tif = ensemble_path+'/CloudImage/', name=file_name,root_address = root_address)   # TIF
        
        save_path_HDF = os.path.join(ensemble_path ,'CloudProduct',file_name+'_CLT.HDF')
        ensemble_result[ensemble_result==0]=255#将第0类转为255
        ensemble_result[ensemble_result==10]=255#将第10类转为255
        save_hdf(ensemble_result,save_path_HDF,1000,1400)
        end_time = time.time()
        print('%d-th image POST-processing time :%d s' % (idx, end_time - start_time))
        start_time = time.time()
        idx = idx + 1

#############################################################################################################################
####################HDFG-->mat#################
def dataloadandsave(fy_path,save_path):
    starttime = time.time()
    fy_list = os.listdir(fy_path)
    savepath_png = save_path +'_png'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(savepath_png)
    # else:
    #     return
    for index in range(len(fy_list)):
        #########风云四号读取数据########
        h5name = os.path.join(fy_path,fy_list[index])
        ########数据处理#########
        geo_range = '5, 55, 70, 140, 0.05'
        #geo_range = '10, 50, 80, 130, 0.05'
        channel = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"]
        warpedimage = correct_GF(h5name,geo_range,channel,1000,1400) #800,1000
        sio.savemat(os.path.join(save_path,'%s.mat'%(fy_list[index][:-4])),{'data':warpedimage})
        #########生成三通道伪彩图#######
        img=warpedimage[:,:,[2,1,0]]*255
        warpedimage_ = Image.fromarray(np.uint8(img))
        warpedimage_.save(os.path.join(savepath_png,'%s.jpg'%(fy_list[index][:-4])))
        print('image {} is saved'.format(index+1))
    endtime = time.time()
    print('HDF-->mat time:%d'%(endtime - starttime))

if __name__ == '__main__':
    start = time.time()
    ensemble_model()
    end = time.time()
    print('total time:%d s'%(end - start))

