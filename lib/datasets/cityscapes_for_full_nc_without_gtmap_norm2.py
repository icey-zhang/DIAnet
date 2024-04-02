# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Jiaqing Zhang & Kai Jiang
# ------------------------------------------------------------------------------
################################################################################
# 2020-11-02
# 1. 对所有模型测nc文件，1000*1400，以图中中国区的1000*1400地区的最大值max和最小值
# min作为归一化的数值
# 2. 对于1000*1400的大图，没有对应的label，只能定性比较判断
################################################################################

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
import scipy.io as sio
from .fy_testdata import correct_GF, correct_GF_800_1000, correct_GF_800_1000_mat, correct_GF_1000_1400_mat

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=False, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(246, 246), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 model_name = None,
                 data_mode = 'HDF',
                 root_HDF_img=''):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)
        assert model_name != None
        self.model_name =  model_name 
        self.data_mode = data_mode
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        self.root_HDF_img = root_HDF_img
        
        #self.img_list = [line.strip().split() for line in open(root+list_path)] #修改

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {0: 0, 1: 1, 2: 2, 3: 3,4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}  #修改
        self.class_weights = torch.FloatTensor([1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000]).cuda() #修改
        #self.class_weights = torch.FloatTensor([0.1, 2.0000, 0.1, 2.0000,  0.1,  0.1, 2.0000,  2.0000,  0.1,2.0000,  0.1]).cuda() #修改
        #self.class_weights = torch.FloatTensor([0.1, 2.0000, 0.1, 2.0000,  0.1,  0.1, 2.0000,  2.0000,  0.1,2.0000,  0.1]).cuda() #修改
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            image_root = '/home/amax/datasets/Himawari864/v2/mattest' #修改  不用
            image_names = os.listdir(image_root)
            for i in range(len(image_names)):
                image_path = os.path.join(image_root, image_names[i])
                name = os.path.splitext(label_names[i])[0]
                print(name)
                files.append({
                    "img": image_path,
                    "name": name,
                })
        elif 'train' in self.list_path:
            #print(self.list_path)

            # if self.data_mode =='HDF':
            #     image_root = r'/home/viewer/em1013/datasets/fy/fy_imagetest_mini' #修改 HDF的路径
            #     label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改
            # elif self.data_mode =='mat':
            #     image_root = '/home/viewer/em1013/datasets/fy/fy_imagetest_mini_mat' #修改 mat的路径 
            #     label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改

            if self.data_mode =='HDF':
                image_root = self.root_HDF_img #修改 HDF的路径
                #label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改 不用
            elif self.data_mode =='mat':
                image_root = self.root_HDF_img + '_mat' #修改 mat的路径 
                #label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改 不用

            image_names = os.listdir(image_root)
            #label_names = os.listdir(label_root)
            for i in range(len(image_names)):
                image_path = os.path.join(image_root, image_names[i])
                image_name_without_jpg = os.path.splitext(image_names[i])[0]
                label_name = image_name_without_jpg + '.mat' #修改
                #label_path = os.path.join(label_root, label_name)
                name = os.path.splitext(label_name)[0]
                #print(image_path)
                #print(label_path)
                #print(name)
                #print('__________________________')
                files.append({
                    "img": image_path,
                    #"label": label_path,
                    "name": name,
                    "weight": 1
                })
        elif 'val' in self.list_path:
            #print(self.list_path)
            
            if self.data_mode =='HDF':
                image_root = self.root_HDF_img #修改 HDF的路径
                #label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改
            elif self.data_mode =='mat':
                image_root = self.root_HDF_img + '_mat'  #修改 mat的路径 
                #label_root = '/home/amax/mnt1/datasets/fy/testv8/matmasktest' #修改

            image_names = os.listdir(image_root)
            #label_names = os.listdir(label_root)
            for i in range(len(image_names)):
                image_path = os.path.join(image_root, image_names[i])
                #image_name_without_jpg = os.path.splitext(image_names[i])[0]
                #label_name = image_name_without_jpg + '.mat' #修改
                #label_path = os.path.join(label_root, label_name)
                name = os.path.splitext(image_names[i])[0]
                #name = os.path.splitext(label_name)[0]
                #print(image_path)
                #print(label_path)
                #print(name)
                #print('__________________________')
                files.append({
                    "img": image_path,
                    #"label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        image_path = item["img"]
        #label_path = item["label"]
        name = item["name"]

        ######### 读取image-nc文件 ##############################
            ########数据处理#########
        geo_range = '5, 55, 70, 140, 0.05'# 标准 1000*1400 高分辨
        #geo_range = '10, 50, 80, 130, 0.1'# 以往 800*1000 低分辨 v2数据集一致
        #geo_range = '5, 55, 70, 140, 0.1'# 目前 800*1000 高分辨 v6 v8 数据集一致
        channel = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"]
        ## 读取1000*1400的图像，以800*1000范围内的最大值和最小值用作归一化，没有label ##
        
        #########数据读取方式改变：两种读取HDF或者mat格式#####
        if self.data_mode == 'HDF':
            #########如果时acw的模型，不做特殊归一化处理，不从800*1000中取最大最小值#####
            if self.model_name[:-2] == 'hrnet_ACW_sub_model':
                image = correct_GF(image_path,geo_range,channel,1000,1400)       
            else:
                image = correct_GF_800_1000(image_path,geo_range,channel,1000,1400)
        elif self.data_mode == 'mat':
            if self.model_name[:-2] == 'hrnet_ACW_sub_model':
                imagemat = sio.loadmat(image_path) #修改
                image = imagemat['data'] #修改
            else:
                #image = correct_GF_800_1000_mat(image_path,channel,1000,1400)
                image = correct_GF_1000_1400_mat(image_path,channel,1000,1400)
        #image = correct_GF_800_1000(image_path,geo_range,channel,1000,1400)
        #image = correct(image_path,geo_range,channel,800,1000)
        if self.model_name != 'nvidia':
            image = self.input_transform(image)#坑啊，通道翻转
        ######## 从1000*1400的图上截取800*1000的图 ##########
        ######## 不从1000*1400的图上截取800*1000的图 ########
        # image = image[100:900,200:1200,:]
        ####################################################
        
        image = image.transpose((2, 0, 1))
        size = image.shape
        #########################################################
        return image.copy(), size, name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
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

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        palette[30:33] = [255,255,255]
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

    def save_pred2(self, preds, sv_path, name):
        palette = self.get_palette(256)
        palette[30:33] = [255,255,255]
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        #label = np.asarray(label.cpu(), dtype=np.uint8).squeeze()
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, ''.join(name)+'.png'))
        """
        label = self.convert_label(label, inverse=True)
        save_img = Image.fromarray(label)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, ''.join(name)+'_label.png'))
        """
