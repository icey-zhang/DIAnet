# -*- coding: utf-8 -*-
"""
从FY-4A标称数据提取指定范围指定通道
Created on 2020/09
@author: zhangjiaqing
"""
from .fy4a import FY4A_AGRI_L1
import scipy.io as sio
import os
import numpy as np
import cv2

import scipy.ndimage
import netCDF4 as nc
import h5py


def fill_ndarray(t1):             # 定义一个函数，把数组中为零的元素替换为一列的均值
    for i in range(t1.shape[1]):
        temp_col = t1[:,i]               # 取出当前列
        nan_num = np.count_nonzero(temp_col != temp_col)    # 判断当前列中是否含nan值
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]            
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()    # 用其余元素的均值填充nan所在位置
    return t1

def correct(fy,geo_range,channel,row,col):
    
    channeldata = np.zeros((row,col,len(channel)))
    fy_4 = np.zeros((row,col,len(channel)))
    s=0
    fy4a_agri_l1 = FY4A_AGRI_L1(fy)
    for i in channel:
        keyword = "Channel" + i
        fy4a_agri_l1.extract(keyword, geo_range)
        ttt = fy4a_agri_l1.channels[keyword][0:row,0:col]
        ttt = fill_ndarray(ttt)
        channeldata[:,:,s] = ttt
        s=s+1
    a = channeldata[:,:,0:6]
    a_nomal = (a-a.min())/(a.max()-a.min())
    b = channeldata[:,:,6:15]
    b_nomal = (b-b.min())/(b.max()-b.min())
    fy_4[:,:,0:6] = a_nomal
    fy_4[:,:,6:15] = b_nomal
    return fy_4

def dataload(fy_path):
    fy_list = os.listdir(fy_path)
    for index in range(len(fy_list)):
        #########风云四号读取数据########
        h5name = os.path.join(fy_path,fy_list[index])
        ########数据处理#########
        geo_range = '5, 55, 70, 140, 0.05'
        channel = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"]
        warpedimage = correct(h5name,geo_range,channel,1000,1400)
        sio.savemat(os.path.join('F:/竞赛云检/fysplit/test1000_1400','%s.mat'%(fy_list[index][:-4])),{'data':warpedimage})
        print('image {} is saved'.format(index+1))

if __name__ == "__main__":
    ##########数据集路径####
    fy_path = 'E:/FY4A/fy_imagetest'  #风云四号数据 HDF 格式
    ##########加载数据#####
    dataload(fy_path)


