Version:0.9 StartHTML:0000000105 EndHTML:0000104774 StartFragment:0000000141 EndFragment:0000104738


import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
import scipy.io as sio

class Cloud38(BaseDataset):
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
                 std=[0.229, 0.224, 0.225]):

        super(Cloud38, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        #self.img_list = [line.strip().split() for line in open(root+list_path)] #修改

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {0: 0, 1: 1, 2: 2, 3: 3,4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}  #修改
        self.class_weights = torch.FloatTensor([1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000, 1.0000,1.0000]).cuda() #修改
    def read_files(self):
        files = []

        if 'train' in self.list_path:

            # image_root = './fysplit/v20/randompick/mattrain' #修改
            # label_root = './fysplit/v20/randompick/matmasktrain' #修改
            image_root = './38-Cloud/38-Cloud_training/train_red/' #修改
            label_root = './38-Cloud/38-Cloud_training/train_gt' #修改
            image_names = os.listdir(image_root)
            # label_names = os.listdir(label_root)
            for i in range(len(image_names)):
                image_path = os.path.join(image_root, image_names[i])
                label_path = os.path.join(label_root, image_names[i].replace('red','gt'))
                name = os.path.splitext(image_names[i])[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        elif 'test' in self.list_path:

            image_root = './38-Cloud/38-Cloud_test/test_red/' #修改
            # label_root = './test/mask' #修改

            image_names = os.listdir(image_root)
            #label_names = os.listdir(label_root)
            for i in range(len(image_names)):
                image_path = os.path.join(image_root, image_names[i])
                # image_name_without_jpg = os.path.splitext(image_names[i])[0]
                # label_name = image_name_without_jpg + '.png' #修改
                # label_path = os.path.join(label_root, label_name)
                name = os.path.splitext(image_names[i])[0]
                #print(image_path)
                #print(label_path)
                #print(name)
                #print('__________________________')
                files.append({
                    "img": image_path,
                    # "label": label_path,
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
        
        name = item["name"]
        # Loading
        r = 'red'
        g = 'green'
        b = 'blue'
        nir = 'nir'
        image_r = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_g = cv2.imread(image_path.replace(r,g), cv2.IMREAD_UNCHANGED)
        image_b = cv2.imread(image_path.replace(r,b), cv2.IMREAD_UNCHANGED)
        image_nir = cv2.imread(image_path.replace(r,nir), cv2.IMREAD_UNCHANGED)
        ## note: with the following transformation for displaying the nir channel is lost at the end ###
        if image_r.shape == image_g.shape == image_b.shape == image_nir.shape:
            # Combining to one img
            image_rgbn = np.dstack((image_r, image_g, image_b, image_nir))
            image = cv2.normalize(image_rgbn, None, 0, 255, cv2.NORM_MINMAX)/255
            
        size = image.shape
        # image = np.asarray(image*255, dtype=np.uint8)
        # save_img = Image.fromarray(image[:,:,0:3])
        # save_img.save(os.path.join('original', ''.join(name)+'.png'))
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name
        image = cv2.resize(image,(192,192)) #如果是训练的话resize
        label_path = item["label"]
        label = cv2.imread((label_path), #修改
                          cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label,(192,192))
        label = label//255 #修改
        label = self.convert_label(label)
        
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
        

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int_(self.crop_size[0] * 1.0)
        stride_w = np.int_(self.crop_size[1] * 1.0)
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
                rows = np.int_(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int_(np.ceil(1.0 * (new_w - 
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

    def save_pred2(self, preds, label, sv_path, name):
        palette = self.get_palette(256)
        palette[30:33] = [255,255,255]
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        label = np.asarray(label.cpu(), dtype=np.uint8).squeeze()
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, ''.join(name)+'.png'))

        label = self.convert_label(label, inverse=True)
        save_img = Image.fromarray(label)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, ''.join(name)+'_label.png'))
        

