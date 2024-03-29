

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
""""""
from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist
from .Fwiou import StreamSegMetrics

import scipy.io as sio
import matplotlib.pyplot as plt
def save_pred_channel_1_as_mat(preds, sv_path, name):
    save_fn = sv_path+'/'+name[0]+'.mat'
    save_array = preds.cpu().numpy().squeeze()
    save_array = save_array.transpose(1,2,0)
    sio.savemat(save_fn, {'data': save_array})

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

####################### 图像剪裁及拼接 ##############################
class Image_Patch_adaptive_plus():
    def __init__(self, image, outChannel, w_stepLen=100, h_stepLen=100):
        """
        Input:
            image: H * W * channel
            outChannel: 1
            w_stepLen: 100
            h_stepLen: 100
        Return:
            B * H * W * Channel
        """
        self.image = image
        self.H, self.W, self.channel = self.image.shape
        self.outChannel = outChannel
        self.w_stepLen = w_stepLen
        self.h_stepLen = h_stepLen

    def to_patch(self):
        sub_w = self.W//4
        sub_h = self.H//4
        self.sub_w = sub_w
        self.sub_h = sub_h
        sub_image_batch = np.zeros((16,self.sub_h+self.h_stepLen,self.sub_w+self.w_stepLen,14))
        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        for i in range(1,17):
            if i == 1:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.h_stepLen+sub_h,      0                       : self.w_stepLen+sub_w,   :]
            elif i == 2:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.h_stepLen+sub_h,      sub_w-self.w_stepLen//2   : 2*sub_w+self.w_stepLen//2, :]
            elif i == 3:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.h_stepLen+sub_h,      2*sub_w-self.w_stepLen//2 : 3*sub_w+self.w_stepLen//2, :]
            elif i == 4:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.h_stepLen+sub_h,      3*sub_w-self.w_stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 5:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.h_stepLen//2  :  2*sub_h+self.h_stepLen//2, 0                       : self.w_stepLen+sub_w,   :]
            elif i == 6:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.h_stepLen//2  :  2*sub_h+self.h_stepLen//2, sub_w-self.w_stepLen//2   : 2*sub_w+self.w_stepLen//2, :]
            elif i == 7:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.h_stepLen//2  :  2*sub_h+self.h_stepLen//2, 2*sub_w-self.w_stepLen//2 : 3*sub_w+self.w_stepLen//2, :]
            elif i == 8:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.h_stepLen//2  :  2*sub_h+self.h_stepLen//2, 3*sub_w-self.w_stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 9:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.h_stepLen//2:  3*sub_h+self.h_stepLen//2, 0                       : self.w_stepLen+sub_w,   :]
            elif i == 10:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.h_stepLen//2:  3*sub_h+self.h_stepLen//2, sub_w-self.w_stepLen//2   : 2*sub_w+self.w_stepLen//2, :]
            elif i == 11:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.h_stepLen//2:  3*sub_h+self.h_stepLen//2, 2*sub_w-self.w_stepLen//2 : 3*sub_w+self.w_stepLen//2, :]
            elif i == 12:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.h_stepLen//2:  3*sub_h+self.h_stepLen//2, 3*sub_w-self.w_stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 13:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.h_stepLen   :  4*sub_h,                 0                       : self.w_stepLen+sub_w,   :]
            elif i == 14:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.h_stepLen   :  4*sub_h,                 sub_w-self.w_stepLen//2   : 2*sub_w+self.w_stepLen//2, :]
            elif i == 15:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.h_stepLen   :  4*sub_h,                 2*sub_w-self.w_stepLen//2 : 3*sub_w+self.w_stepLen//2, :]
            elif i == 16:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.h_stepLen   :  4*sub_h,                 3*sub_w-self.w_stepLen    : 4*sub_w, :]
        self.sub_image_batch = sub_image_batch
        return self.sub_image_batch
    def recover(self, pred_image_batch):
        """
        Input:
            B * H * W * Channel
        Return:
            image: H * W * channel
            outChannel: 1
            stepLen: 100
        """
        self.recover_image = np.zeros([self.H,self.W,self.outChannel])
        def rec(pred_image_batch):
            try:
                self.recover_image[((i-1)//4)*self.sub_h:((i-1)//4+1)*self.sub_h,((i-1)%4)*self.sub_w:((i-1)%4+1)*self.sub_w,:] = pred_image_batch
            except:
                print(i)
                print('ERROR!!!!!!!!!!!!!!!')
        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        #300*350
        #200*250
        for i in range(1,17):
            if i == 1:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 0               : self.sub_w,                    :])
            elif i == 4:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 self.w_stepLen    : self.w_stepLen+self.sub_w,       :])
            elif i == 13:
                rec(pred_image_batch[i-1, self.h_stepLen    : self.h_stepLen+self.sub_h,    0               : self.sub_w,                    :])
            elif i == 16:
                rec(pred_image_batch[i-1, self.h_stepLen    : self.h_stepLen+self.sub_h,    self.w_stepLen    : self.w_stepLen+self.sub_w,       :])
            elif i == 2 or i == 3:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 self.w_stepLen//2 : self.w_stepLen//2+self.sub_w,    :])
            elif i == 14 or i == 15:
                rec(pred_image_batch[i-1, self.h_stepLen    : self.h_stepLen+self.sub_h,    self.w_stepLen//2 : self.w_stepLen//2+self.sub_w,    :])
            elif i ==5 or i == 9:
                rec(pred_image_batch[i-1, self.h_stepLen//2 : self.h_stepLen//2+self.sub_h, 0               : self.sub_w,                    :])
            elif i ==8 or i == 12:
                rec(pred_image_batch[i-1, self.h_stepLen//2 : self.h_stepLen//2+self.sub_h, self.w_stepLen    : self.w_stepLen+self.sub_w,       :])
            elif i ==6 or i == 7 or i == 10 or i == 11:
                rec(pred_image_batch[i-1, self.h_stepLen//2 : self.h_stepLen//2+self.sub_h, self.w_stepLen//2 : self.w_stepLen//2+self.sub_w,    :])

        return self.recover_image
    
####################################################################


####################### 图像剪裁及拼接 ##############################
class Image_Patch_adaptive():
    def __init__(self, image, outChannel, stepLen=100):
        """
        Input:
            image: H * W * channel
            outChannel: 1
            stepLen: 100
        Return:
            B * H * W * Channel
        """
        self.image = image
        self.H, self.W, self.channel = self.image.shape
        self.outChannel = outChannel
        self.stepLen = stepLen
    def to_patch(self):
        sub_w = self.W//4
        sub_h = self.H//4
        self.sub_w = sub_w
        self.sub_h = sub_h
        sub_image_batch = np.zeros((16,self.sub_h+self.stepLen,self.sub_w+self.stepLen,14))
        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        for i in range(1,17):
            if i == 1:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.stepLen+sub_h,      0                       : self.stepLen+sub_w,   :]
            elif i == 2:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.stepLen+sub_h,      sub_w-self.stepLen//2   : 2*sub_w+self.stepLen//2, :]
            elif i == 3:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.stepLen+sub_h,      2*sub_w-self.stepLen//2 : 3*sub_w+self.stepLen//2, :]
            elif i == 4:
                sub_image_batch[i-1,:,:,:] = self.image[0                      :  self.stepLen+sub_h,      3*sub_w-self.stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 5:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.stepLen//2  :  2*sub_h+self.stepLen//2, 0                       : self.stepLen+sub_w,   :]
            elif i == 6:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.stepLen//2  :  2*sub_h+self.stepLen//2, sub_w-self.stepLen//2   : 2*sub_w+self.stepLen//2, :]
            elif i == 7:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.stepLen//2  :  2*sub_h+self.stepLen//2, 2*sub_w-self.stepLen//2 : 3*sub_w+self.stepLen//2, :]
            elif i == 8:
                sub_image_batch[i-1,:,:,:] = self.image[sub_h-self.stepLen//2  :  2*sub_h+self.stepLen//2, 3*sub_w-self.stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 9:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.stepLen//2:  3*sub_h+self.stepLen//2, 0                       : self.stepLen+sub_w,   :]
            elif i == 10:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.stepLen//2:  3*sub_h+self.stepLen//2, sub_w-self.stepLen//2   : 2*sub_w+self.stepLen//2, :]
            elif i == 11:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.stepLen//2:  3*sub_h+self.stepLen//2, 2*sub_w-self.stepLen//2 : 3*sub_w+self.stepLen//2, :]
            elif i == 12:
                sub_image_batch[i-1,:,:,:] = self.image[2*sub_h-self.stepLen//2:  3*sub_h+self.stepLen//2, 3*sub_w-self.stepLen    : 4*sub_w, :]
            #########################################################################################################################
            elif i == 13:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.stepLen   :  4*sub_h,                 0                       : self.stepLen+sub_w,   :]
            elif i == 14:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.stepLen   :  4*sub_h,                 sub_w-self.stepLen//2   : 2*sub_w+self.stepLen//2, :]
            elif i == 15:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.stepLen   :  4*sub_h,                 2*sub_w-self.stepLen//2 : 3*sub_w+self.stepLen//2, :]
            elif i == 16:
                sub_image_batch[i-1,:,:,:] = self.image[3*sub_h-self.stepLen   :  4*sub_h,                 3*sub_w-self.stepLen    : 4*sub_w, :]
        self.sub_image_batch = sub_image_batch
        return self.sub_image_batch
    def recover(self, pred_image_batch):
        """
        Input:
            B * H * W * Channel
        Return:
            image: H * W * channel
            outChannel: 1
            stepLen: 100
        """
        self.recover_image = np.zeros([self.H,self.W,self.outChannel])
        def rec(pred_image_batch):
            try:
                self.recover_image[((i-1)//4)*self.sub_h:((i-1)//4+1)*self.sub_h,((i-1)%4)*self.sub_w:((i-1)%4+1)*self.sub_w,:] = pred_image_batch
            except:
                print(i)
                print('ERROR!!!!!!!!!!!!!!!')
        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        #300*350
        #200*250
        for i in range(1,17):
            if i == 1:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 0               : self.sub_w,                    :])
            elif i == 4:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 self.stepLen    : self.stepLen+self.sub_w,       :])
            elif i == 13:
                rec(pred_image_batch[i-1, self.stepLen    : self.stepLen+self.sub_h,    0               : self.sub_w,                    :])
            elif i == 16:
                rec(pred_image_batch[i-1, self.stepLen    : self.stepLen+self.sub_h,    self.stepLen    : self.stepLen+self.sub_w,       :])
            elif i == 2 or i == 3:
                rec(pred_image_batch[i-1, 0               : self.sub_h,                 self.stepLen//2 : self.stepLen//2+self.sub_w,    :])
            elif i == 14 or i == 15:
                rec(pred_image_batch[i-1, self.stepLen    : self.stepLen+self.sub_h,    self.stepLen//2 : self.stepLen//2+self.sub_w,    :])
            elif i ==5 or i == 9:
                rec(pred_image_batch[i-1, self.stepLen//2 : self.stepLen//2+self.sub_h, 0               : self.sub_w,                    :])
            elif i ==8 or i == 12:
                rec(pred_image_batch[i-1, self.stepLen//2 : self.stepLen//2+self.sub_h, self.stepLen    : self.stepLen+self.sub_w,       :])
            elif i ==6 or i == 7 or i == 10 or i == 11:
                rec(pred_image_batch[i-1, self.stepLen//2 : self.stepLen//2+self.sub_h, self.stepLen//2 : self.stepLen//2+self.sub_w,    :])

        return self.recover_image
    
####################################################################

####################### 图像剪裁及拼接 ##############################
class Image_Patch():
    """
    输入：行，列，通道
    H = 800
    W = 1000
    """
    def __init__(self, image):
        self.image = image
    def to_patch(self):
        W,H,Dim = self.image.shape
        sub_w = W//4
        sub_h = H//4
        sub_image_batch = np.zeros((16,sub_w+100,sub_h+100,14))

        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        #300*350
        for i in range(1,17):
            if i == 1:
                sub_image_batch[i-1,:,:,:] = self.image[0:300,0:350, :]
            elif i == 4:
                sub_image_batch[i-1,:,:,:] = self.image[0:300,650:1000, :]
            elif i == 13:
                sub_image_batch[i-1,:,:,:] = self.image[500:800,0:350, :]
            elif i == 16:
                sub_image_batch[i-1,:,:,:] = self.image[500:800,650:1000, :]
            elif i == 2 or i == 3:
                if i == 2:
                    sub_image_batch[i-1,:,:,:] = self.image[0:300,200:550, :]
                elif i == 3:
                    sub_image_batch[i-1,:,:,:] = self.image[0:300,450:800, :]
            elif i == 14 or i == 15:
                if i == 14:
                    sub_image_batch[i-1,:,:,:] = self.image[500:800,200:550, :]
                elif i == 15:
                    sub_image_batch[i-1,:,:,:] = self.image[500:800,450:800, :]
            elif i ==5 or i == 9:
                if i == 5:
                    sub_image_batch[i-1,:,:,:] = self.image[150:450,0:350, :]
                elif i == 9:
                    sub_image_batch[i-1,:,:,:] = self.image[350:650,0:350, :]
            elif i ==8 or i == 12:
                if i == 8:
                    sub_image_batch[i-1,:,:,:] = self.image[150:450,650:1000, :]
                elif i == 12:
                    sub_image_batch[i-1,:,:,:] = self.image[350:650,650:1000, :]
            elif i ==6 or i == 7 or i == 10 or i == 11:
                if i == 6:
                    sub_image_batch[i-1,:,:,:] = self.image[150:450,200:550, :]
                elif i == 7:
                    sub_image_batch[i-1,:,:,:] = self.image[150:450,450:800, :]
                elif i == 10:
                    sub_image_batch[i-1,:,:,:] = self.image[350:650,200:550, :]
                elif i == 11:
                    sub_image_batch[i-1,:,:,:] = self.image[350:650,450:800, :]
        self.sub_image_batch = sub_image_batch
        return sub_image_batch
    def recover(self, pred_image_batch):
        self.recover_image = np.zeros([800,1000,11])
        def rec(pred_image_batch):
            try:
                self.recover_image[((i-1)//4)*200:((i-1)//4+1)*200,((i-1)%4)*250:((i-1)%4+1)*250,:] = pred_image_batch
            except:
                print(i)
                print('ERROR!!!!!!!!!!!!!!!')
        """
        1  2  3  4
        5  6  7  8
        9  10 11 12
        13 14 15 16
        """
        #300*350
        #200*250
        for i in range(1,17):
            if i == 1:
                rec(pred_image_batch[i-1,0:200,0:250,:])
            elif i == 4:
                rec(pred_image_batch[i-1,0:200,100:350,:])
            elif i == 13:
                rec(pred_image_batch[i-1,100:300,0:250,:])
            elif i == 16:
                rec(pred_image_batch[i-1,100:300,100:350,:])
            elif i == 2 or i == 3:
                if i == 2:
                    rec(pred_image_batch[i-1,0:200,50:300,:])
                elif i == 3:
                    rec(pred_image_batch[i-1,0:200,50:300,:])
            elif i == 14 or i == 15:
                if i == 14:
                    rec(pred_image_batch[i-1,100:300,50:300,:])
                elif i == 15:
                    rec(pred_image_batch[i-1,100:300,50:300,:])
            elif i ==5 or i == 9:
                if i == 5:
                    rec(pred_image_batch[i-1,50:250,0:250,:])
                elif i == 9:
                    rec(pred_image_batch[i-1,50:250,0:250,:])
            elif i ==8 or i == 12:
                if i == 8:
                    rec(pred_image_batch[i-1,50:250,100:350,:])
                elif i == 12:
                    rec(pred_image_batch[i-1,50:250,100:350,:])
            elif i ==6 or i == 7 or i == 10 or i == 11:
                if i == 6:
                    rec(pred_image_batch[i-1,50:250,50:300,:])
                elif i == 7:
                    rec(pred_image_batch[i-1,50:250,50:300,:])
                elif i == 10:
                    rec(pred_image_batch[i-1,50:250,50:300,:])
                elif i == 11:
                    rec(pred_image_batch[i-1,50:250,50:300,:])

        return self.recover_image
    


def changeto6classfrom11class(label, result):
    ##############改变类别编号#############
    label_ = np.zeros(label.shape)
    result_ = np.zeros(result.shape)
    label_[label==10]=5
    label_[label==2]=5
    label_[label==4]=5
    # label[label==5]=5
    label_[label==8]=5
    label_[label==0]=5

    label_[label==6]=0
    label_[label==9]=1
    label_[label==7]=2
    label_[label==1]=3
    label_[label==3]=4

    result_[result==10]=5
    result_[result==2]=5
    result_[result==4]=5
    # result[result==5]=5
    result_[result==8]=5
    result_[result==0]=5

    result_[result==6]=0
    result_[result==9]=1
    result_[result==7]=2
    result_[result==1]=3
    result_[result==3]=4
    return label_,result_


####################################################################

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    
    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, FwIoU

def validate_patch(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            #losses, pred = model(image, label)
            
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=50, h_stepLen=50)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, FwIoU

def validate_patch_for_hr_ocr(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            #losses, pred = model(image, label)
            
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=50, h_stepLen=50)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                # pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                out_aux_seg = model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:])
                pred.append(out_aux_seg[1])
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, FwIoU


def validate2_patch_for_hr_ocr(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                out_aux_seg = model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:])
                pred.append(out_aux_seg[1])
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, FwIoU


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():

        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = '/root/HRNet/results'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)


def validate2(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def validate2_cpu(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image
            label = label.long()

            #losses, pred = model(image, label)
            pred = model(image)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2_/prob/'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, './results2_/', name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)
            """
            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())
            """
    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix)
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    #writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return mean_IoU, IoU_array
    #return ave_loss.average(), mean_IoU, IoU_array

def validate2_cpu_nc(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    #metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        #metrics.reset()
        for idx, batch in enumerate(testloader):
            image, size, name = batch
            #image = image.transpose(0,3,1,2)#.cuda()
            #image = torch.from_numpy(image.numpy().transpose(0,3,1,2)).float()
            for i in range(14):
                plt.figure()
                plt.imshow(image[0,i,:,:].numpy().squeeze())
                plt.savefig(str(i)+'1.png')

            #losses, pred = model(image, label)
            pred = model(image)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                #metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2_/prob/'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, sv_path, name)
    return 0
    #return ave_loss.average(), mean_IoU, IoU_array

def validate2_patch(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                try:
                    print('#################################')
                    print('####### model.module.model ######')
                    print('#################################')
                    pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                
                except:
                    print('#################################')
                    print('############## model ############')
                    print('#################################')
                    pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                    os.mkdir(sv_path+'/prob/')
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.get_results())
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def validate2_nc_patch(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=50, h_stepLen=50)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

def validate2_mini_patch(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=250, h_stepLen=200)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred = []
            for sub_image_index in range(16):
                #########切块mini_patch##############
                #image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                mini_image_patch = Image_Patch_adaptive_plus(image=patch[sub_image_index:sub_image_index+1,:,:,:].cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=50, h_stepLen=50)
                mini_patch = torch.from_numpy(mini_image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred_mini_image_patch=[]
                for sub_mini_image_index in range(16):
                    pred_mini_image_patch.append(model.module.model(mini_patch[sub_mini_image_index:sub_mini_image_index+1,:,:,:]))
                pred_mini_image_patch = torch.cat(pred_mini_image_patch)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred_mini_image_patch = torch.from_numpy(np.expand_dims(mini_image_patch.recover(pred_mini_image_patch.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
                ###############################################
                pred.append(pred_mini_image_patch)
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

def validate2_patch_ensemble(config, test_dataset, testloader, model, writer_dict, sub_model_name=None):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                if sub_model_name == 'hrnet_ocr':
                    pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])
                else:
                    pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './save_mat/'+sub_model_name+'results2'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                    test_dataset.save_pred2(x, label, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                else:
                    print('\nThe predicted results are not saved\n')

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


#调用的这个函数
def validate2_patch_ensemble_for_nc(config, test_dataset, testloader, model, writer_dict, sub_model_name=None, patch_mode='on',root_address=''):# 加
    # 测nc高分辨图像，从1000*1400的图上截取相应的800*1000的图，
    # 有label，
    # 计算IoU，
    # 保存png，mat两种文件
    # 保存的文件后缀为：'_ncdata_matlabel_results2_norm1'
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, size, name = batch
            #size = label.size()
            image = image.cuda()                # [1, 14, 800, 1000]
            label = label.long().cuda()
            
            #print(idx)
            
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                # image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                for sub_image_index in range(16):
                    #pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])) #标记一下
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])        # 一个一个输入网络,hrnet里面输出aux和pred
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = root_address + '/tools/'+sub_model_name+'_ncdata_matlabel_results2_norm4'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                    test_dataset.save_pred2(x, label, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                else:
                    print('\nThe predicted results are not saved\n')

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    #writer = writer_dict['writer']
    #global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    #writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    #writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    #writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array



#调用的这个函数
def validate2_patch_ensemble_for_nc_900_1000(config, test_dataset, testloader, model, writer_dict, sub_model_name=None, patch_mode='on',root_address=''):# 加
    # 测nc1000*1400高分辨图像，
    # 从1000*1400的结果上截取相应的900*1000的图，
    # 有label，
    # 计算IoU，
    # 保存png，mat两种文件
    # 保存的文件后缀为：'_ncdata_matlabel_results2_norm1'
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, size, name = batch
            #size = label.size()
            image = image.cuda()                # [1, 14, 800, 1000]
            label = label.long().cuda()
            
            #print(idx)
            
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                # image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                for sub_image_index in range(16):
                    #pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])) #标记一下
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])        # 一个一个输入网络,hrnet里面输出aux和pred
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                ######## 截取 900*1000 的区域用于计算精度 ###################################
                x = x[:,:,100:1000,300:1300]
                ############################################################################
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = root_address + '/tools/'+sub_model_name+'_ncdata_matlabel_results2_norm6'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                    test_dataset.save_pred2(x, label, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                else:
                    print('\nThe predicted results are not saved\n')

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    #writer = writer_dict['writer']
    #global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    #writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    #writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    #writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

def validate2_patch_ensemble_for_nc_900_1000_6class(config, test_dataset, testloader, model, writer_dict, sub_model_name=None, patch_mode='on',root_address=''):# 加
    # 测nc1000*1400高分辨图像，
    # 从1000*1400的结果上截取相应的900*1000的图，
    # 有label，
    # 计算IoU，
    # 保存png，mat两种文件
    # 保存的文件后缀为：'_ncdata_matlabel_results2_norm1'
    # 将原来的11类转变成五类计算
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(6) #五类和其他
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, size, name = batch
            #size = label.size()
            image = image.cuda()                # [1, 14, 800, 1000]
            label = label.long().cuda()
            
            #print(idx)
            
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                # image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                for sub_image_index in range(16):
                    #pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])) #标记一下
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])        # 一个一个输入网络,hrnet里面输出aux和pred
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                ######## 截取 900*1000 的区域用于计算精度 ###################################
                x = x[:,:,100:1000,300:1300]
                ############################################################################
                c_10_semantic_image = label.cpu().numpy()
                fy_4_ = x.detach().max(dim=1)[1].cpu().numpy()
                label_change,result_change = changeto6classfrom11class(c_10_semantic_image, fy_4_)
                metrics.update(np.uint8(label_change), np.uint8(result_change))
                sv_path = root_address + '/tools/'+sub_model_name+'_ncdata_matlabel_results2_norm6'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                    test_dataset.save_pred2(x, label, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                else:
                    print('\nThe predicted results are not saved\n')

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    #writer = writer_dict['writer']
    #global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    #writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    #writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    #writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


####没用########
def validate2_patch_ensemble_for_nc_cpu(config, test_dataset, testloader, model, writer_dict, sub_model_name=None): 
    model.eval()
    
    with torch.no_grad():

        for idx, batch in enumerate(testloader):
            image, size, name = batch

            image = image#.cuda()      # [1, 14, 800, 1000]
            #########切块patch##############
            # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
            image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float()#.cuda()   # [16, 14, 300, 350]
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                if sub_model_name == 'hrnet_ocr':
                    pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])        # 一个一个输入网络,hrnet里面输出aux和pred
                else:
                    pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)
            ###############################################
            

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                sv_path = './'+sub_model_name+'_nc_results2'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                    test_dataset.save_pred2(x, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                else:
                    print('\nThe predicted results are not saved\n')

    return 0


def validate2_patch_ensemble_for_nc_gpu(config, test_dataset, testloader, model, writer_dict, sub_model_name=None,root_address='',patch_mode='on'):
    # 测nc高分辨图像
    # 无label
    # 保存tif，png，mat三种文件
    model.eval()
    
    with torch.no_grad():

        for idx, batch in enumerate(testloader):
            image, size, name = batch

            image = image.cuda()
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                for sub_image_index in range(16):
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:])[0])
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                    #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                sv_path = root_address + '/tools/'+sub_model_name+'_nc_results2//'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                        # os.mkdir(sv_path+'/CloudImage/')
                        # os.mkdir(sv_path+'/CloudProduct/')
                    test_dataset.save_pred2(x, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                    # scatter_Asian(mask = np.argmax(x.numpy(),axis=1).squeeze(),save_path_tif = sv_path+'/CloudImage/', name=name[0],root_address=root_address)
                    
                    # save_path_HDF = os.path.join(sv_path ,'CloudProduct',name[0]+'_CLT.HDF')
                    # label_transform = np.argmax(x.numpy(),axis=1).squeeze()
                    # label_transform[label_transform==10]=255
                    # label_transform[label_transform==0]=255
                    # save_hdf(label_transform,save_path_HDF,1000,1400)

                    #scatter_Asian(mask = label_transform,save_path_tif = sv_path+'/TIF/', name=name[0])

                else:
                    print('\nThe predicted results are not saved\n')

    return 0

def validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive(config, test_dataset, testloader, model, writer_dict, sub_model_name=None,root_address='',patch_mode='on',sub_batch_size=1):
    # 测nc高分辨图像
    # 无label
    # 保存tif，png，mat三种文件
    model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        for idx, batch in enumerate(testloader):


            image, size, name = batch

            ##
            image = image.cuda()
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                #for sub_image_index in range(16):
                for sub_image_index in range(0,16,sub_batch_size):
                    #print('\nsub_image_index')
                    #print(sub_image_index)
                    #print('\nsub_batch_size')
                    #print(sub_batch_size)
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+sub_batch_size,:,:,:])[0])
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+sub_batch_size,:,:,:]))
                    #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            end_time = time.time()
            print('%d-th image processing time :%d s' % (idx, end_time - start_time))
            start_time = time.time()
            ###加：patch修改###

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                sv_path = root_address + '/tools/'+sub_model_name+'_nc_results2//'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                        # os.mkdir(sv_path+'/CloudImage/')
                        # os.mkdir(sv_path+'/CloudProduct/')
                    test_dataset.save_pred2(x, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                    # scatter_Asian(mask = np.argmax(x.numpy(),axis=1).squeeze(),save_path_tif = sv_path+'/CloudImage/', name=name[0],root_address=root_address)
                    
                    # save_path_HDF = os.path.join(sv_path ,'CloudProduct',name[0]+'_CLT.HDF')
                    # label_transform = np.argmax(x.numpy(),axis=1).squeeze()
                    # label_transform[label_transform==10]=255
                    # label_transform[label_transform==0]=255
                    # save_hdf(label_transform,save_path_HDF,1000,1400)

                    #scatter_Asian(mask = label_transform,save_path_tif = sv_path+'/TIF/', name=name[0])

                else:
                    print('\nThe predicted results are not saved\n')

    return 0

#validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive_for_awca_psnl
def validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive_for_awca_psnl(config, test_dataset, testloader, model, writer_dict, sub_model_name=None,root_address='',patch_mode='on',sub_batch_size=1):
    # awca_psnl模块的模型测试
    # 测nc高分辨图像
    # 无label
    # 保存tif，png，mat三种文件
    model.eval()
    
    with torch.no_grad():

        for idx, batch in enumerate(testloader):
            image, size, name = batch

            image = image.cuda()
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=110, h_stepLen=110)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                #for sub_image_index in range(16):
                for sub_image_index in range(0,16,1):
                    #print('\nsub_image_index')
                    #print(sub_image_index)
                    #print('\nsub_batch_size')
                    #print(sub_batch_size)
                    sub_image_patch = Image_Patch_adaptive_plus(image=patch[sub_image_index:sub_image_index+1,:,:,:].cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=40, h_stepLen=40)
                    sub_patch = torch.from_numpy(sub_image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                    sub_pred=[]
                    for sub_sub_image_index in range(0,16,sub_batch_size):
                        if sub_model_name == 'hrnet_ocr':
                            sub_pred.append(model(sub_patch[sub_sub_image_index:sub_sub_image_index+sub_batch_size,:,:,:])[0])
                        else:
                            sub_pred.append(model(sub_patch[sub_sub_image_index:sub_sub_image_index+sub_batch_size,:,:,:]))
                    #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                    sub_pred = torch.cat(sub_pred)
                    sub_pred = torch.from_numpy(np.expand_dims(sub_image_patch.recover(sub_pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)

                    pred.append(sub_pred)
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                sv_path = root_address + '/tools/'+sub_model_name+'_nc_results2//'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                        # os.mkdir(sv_path+'/CloudImage/')
                        # os.mkdir(sv_path+'/CloudProduct/')
                    test_dataset.save_pred2(x, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                    # scatter_Asian(mask = np.argmax(x.numpy(),axis=1).squeeze(),save_path_tif = sv_path+'/CloudImage/', name=name[0],root_address=root_address)
                    
                    # save_path_HDF = os.path.join(sv_path ,'CloudProduct',name[0]+'_CLT.HDF')
                    # label_transform = np.argmax(x.numpy(),axis=1).squeeze()
                    # label_transform[label_transform==10]=255
                    # label_transform[label_transform==0]=255
                    # save_hdf(label_transform,save_path_HDF,1000,1400)

                    #scatter_Asian(mask = label_transform,save_path_tif = sv_path+'/TIF/', name=name[0])

                else:
                    print('\nThe predicted results are not saved\n')

    return 0

def validate2_patch_ensemble_for_nc_gpu_sub_batch_size_adaptive_for_nvidia(config, test_dataset, testloader, model, writer_dict, sub_model_name=None,root_address='',patch_mode='on',sub_batch_size=1):
    # 测nc高分辨图像
    # 无label
    # 保存tif，png，mat三种文件
    model.eval()
    
    with torch.no_grad():

        for idx, batch in enumerate(testloader):
            image, size, name = batch

            image = image.cuda()
            ###加：patch修改###
            if patch_mode == 'on':
                #########切块patch##############
                # image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
                #image_patch = Image_Patch_adaptive(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,stepLen=100)
                image_patch = Image_Patch_adaptive_plus(image=image.cpu().numpy().transpose(0,2,3,1).squeeze(), outChannel=11,w_stepLen=100, h_stepLen=100)
                patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
                ###############################################
                pred=[]
                #for sub_image_index in range(16):
                for sub_image_index in range(0,16,sub_batch_size):
                    #print('\nsub_image_index')
                    #print(sub_image_index)
                    #print('\nsub_batch_size')
                    #print(sub_batch_size)
                    if sub_model_name == 'hrnet_ocr':
                        pred.append(model(patch[sub_image_index:sub_image_index+sub_batch_size,:,:,:])[0])
                    elif sub_model_name == 'nvidia':
                        input_dict = {'images':patch[sub_image_index:sub_image_index+sub_batch_size,:,:,:]}
                        output_dict = model(input_dict)
                        pred.append(output_dict['pred'])
                    else:
                        pred.append(model(patch[sub_image_index:sub_image_index+sub_batch_size,:,:,:]))
                    #pred.append(model.module.model(patch[sub_image_index:sub_image_index+1,:,:,:]))
                pred = torch.cat(pred)
                losses = np.zeros(3)
                #########切块patch、拼起来##############
                pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().permute(0,3,1,2)
                ###############################################
            elif patch_mode == 'off':
                pred = model(image)
            ###加：patch修改###

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                sv_path = root_address + '/tools/'+sub_model_name+'_nc_results2//'

                if sub_model_name != None:
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                        os.mkdir(sv_path+'/prob/')
                        # os.mkdir(sv_path+'/CloudImage/')
                        # os.mkdir(sv_path+'/CloudProduct/')
                    test_dataset.save_pred2(x, sv_path, name)
                    save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
                    # scatter_Asian(mask = np.argmax(x.numpy(),axis=1).squeeze(),save_path_tif = sv_path+'/CloudImage/', name=name[0],root_address=root_address)
                    
                    # save_path_HDF = os.path.join(sv_path ,'CloudProduct',name[0]+'_CLT.HDF')
                    # label_transform = np.argmax(x.numpy(),axis=1).squeeze()
                    # label_transform[label_transform==10]=255
                    # label_transform[label_transform==0]=255
                    # save_hdf(label_transform,save_path_HDF,1000,1400)

                    #scatter_Asian(mask = label_transform,save_path_tif = sv_path+'/TIF/', name=name[0])

                else:
                    print('\nThe predicted results are not saved\n')

    return 0

def validate2_patch_for_hr_psp(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            for sub_image_index in range(16):
                pred.append(model(patch[sub_image_index:sub_image_index+1,:,:,:]))
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array\

def validate2_patch_with_interpolate(config, test_dataset, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    metrics = StreamSegMetrics(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        metrics.reset()
        for idx, batch in enumerate(testloader):
            image, label, _, name, *border_padding = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            #########切块patch##############
            image_patch = Image_Patch(image.cpu().numpy().transpose(0,2,3,1).squeeze())
            patch = torch.from_numpy(image_patch.to_patch().transpose(0,3,1,2)).float().cuda()
            ###############################################
            pred=[]
            interpolated_pred=[]
            for sub_image_index in range(16):
                ######## interpolate 6:7 #################################################
                interpolated_patch = F.interpolate(
                    input=patch[sub_image_index:sub_image_index+1,:,:,:], size=[480,560],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                ######## prediction ########################################################
                interpolated_pred.append(model.module.model(interpolated_patch))
                ######## down_sampling 6:7 #################################################
                for patch_index in range(16):
                    pred.append(F.interpolate(
                    input=interpolated_pred[patch_index:patch_index+1,:,:,:], size=[300,350],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                    ))
                ######## down_sampling 6:7 #################################################
            pred = torch.cat(pred)
            losses = np.zeros(3)
            #########切块patch、拼起来##############
            pred = torch.from_numpy(np.expand_dims(image_patch.recover(pred.cpu().numpy().transpose(0,2,3,1)),0)).float().cuda().permute(0,3,1,2)
            ###############################################
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                metrics.update(label.cpu().numpy(),x.detach().max(dim=1)[1].cpu().numpy())
                sv_path = './results2'
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(x, label, sv_path, name)


                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            #if idx % 10 == 0:
            #    print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    score = metrics.get_results()
    print(metrics.to_str(score))
    FwIoU = score['fwIoU']
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid_FwIoU', FwIoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def validate_ensemble_for_nc_gpu(args, val_loader, net, sub_model_name=None):
    '''
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch 
    writer: tensorboard writer
    return: 
    '''
    net.eval()
    for vi, data in enumerate(val_loader):
        input, _, name = data


        input.cuda()

        with torch.no_grad():
            seg_out, _ = net(input)    # output = (1, 19, 713, 713)

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions

        #seg_predictions = seg_out.data.max(1)[1].cpu()
        seg_pred = seg_out.data.cpu().numpy()
        x = seg_pred.transpose(0,2,3,1)

        #Logging
        if vi % 20 == 0:
            if args.local_rank == 0:
                logging.info('validating: %d / %d' % (vi + 1, len(val_loader)))
    
        sv_path = './'+sub_model_name+'_nc_results2'

        if sub_model_name != None:
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
                os.mkdir(sv_path+'/prob/')
            save_pred_channel_1_as_mat(preds=(x), sv_path=sv_path+'/prob/'+'/', name=name)
            #scatter_Asian(mask = np.argmax(x.numpy(),axis=1).squeeze(),save_path_tif = sv_path+'/TIF/', name=name[0])
            #save_path_HDF = os.path.join(sv_path ,'HDF',name[0]+'_CLT.HDF')
            #save_hdf(np.argmax(x.numpy(),axis=1).squeeze(),save_path_HDF,1000,1400)
        else:
            print('\nThe predicted results are not saved\n')


if __name__ == '__main__':
    """
    for i in range(1,17):
        print('############')
        print(i//4*200)
        #print((i//4+1)*250)
        
        print(i%4*250)
        #print((i%4+1)*250)
    """
    # 1000 * 1400
    path = '/data/xieweiying/dataset/fy/test1000_1400/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200607033000_20200607033417_4000M_V0001.mat'
    labelmat = sio.loadmat(path) #修改
    image = labelmat['data'] #修改
    #image = image.reshape([1,800,1000,14])
    image_patch = Image_Patch_adaptive_plus(image=image, outChannel=14,w_stepLen=100, h_stepLen=50)
    patch = image_patch.to_patch()
    #####################################
    for sub_image_index in range(16):
        mini_image = Image_Patch_adaptive_plus(image=patch[sub_image_index:sub_image_index+1,:,:,:].squeeze(), outChannel=14,w_stepLen=50, h_stepLen=50)
        mini_patch = mini_image.to_patch()
        rec_mini = mini_image.recover(mini_patch)
    #####################################
    rec = image_patch.recover(patch)
    plt.figure()
    plt.imshow(image[:,:,0].squeeze())
    plt.savefig('ori.png')
    plt.figure()
    plt.imshow(rec[:,:,0].squeeze())
    plt.savefig('rec.png')

