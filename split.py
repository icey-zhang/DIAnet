###########说明############
#选取的区域为5-50，85-135
#根据区域范围，将1000*1400的图做归一化取出900*1000
#将图裁成100*100的大小
#如果风云的拍摄时间段包含葵花八号的，就认为两数据对应，比如风云时间是0238-0242，则认为0240的葵花八号数据是对应标签
from fy4a import FY4A_AGRI_L1
import scipy.io as sio
import os
import numpy as np
import cv2  as cv
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import scipy.ndimage
from netCDF4 import Dataset
from PIL import Image
from numpy.matlib import random
def nctomatmask(nc_label_path):
    nc_obj = Dataset(nc_label_path)
    type = nc_obj.variables.keys()
    keyword = "CLTYPE"
    for k in type:
        if str(k).find(keyword) == 0:
            data = nc_obj.variables[k][:]
            mask = data.filled() # data的类型是masked_array
            mask[mask==255]=10
    return mask

def fill_ndarray(t1):             # 定义一个函数，把数组中为零的元素替换为一列的均值
    for i in range(t1.shape[1]):
        temp_col = t1[:,i]               # 取出当前列
        nan_num = np.count_nonzero(temp_col != temp_col)    # 判断当前列中是否含nan值
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]            
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()    # 用其余元素的均值填充nan所在位置
    return t1

def changecolor(img):
    row,col = img.shape
    I=np.zeros((row,col),dtype=np.uint8)
    new_img = cv.cvtColor(I,cv.COLOR_GRAY2RGB)
    # 此处替换颜色，为BGR通道
    new_img[img == 1] = (204, 204, 255)
    new_img[img == 2] = (102, 102, 255)
    new_img[img == 3] = (0, 0, 255)
    new_img[img == 4] = (255, 255, 204)
    new_img[img == 5] = (255,255,0)
    new_img[img == 6] = (204,204,0)
    new_img[img == 7] = (255,153,153)
    new_img[img == 8] = (255, 102, 51)
    new_img[img == 9] = (255, 0, 0)
    new_img[img == 10] = (255, 255, 255)
    new_img[img == 0] = (0, 0, 0)
    return new_img

def correct(fy,label):
    channel = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"]
    channeldata = np.zeros((1000,1400,14))
    fy_4 = np.zeros((1000,1400,14))
    s=0
    fy4a_agri_l1 = FY4A_AGRI_L1(fy)
    for i in channel:
        keyword = "Channel" + i
        geo_range = '5, 55, 70, 140, 0.05'
        fy4a_agri_l1.extract(keyword, geo_range)
        ttt = fy4a_agri_l1.channels[keyword][0:1000,0:1400]
        ttt = fill_ndarray(ttt)
        channeldata[:,:,s] = ttt
        s=s+1
    a = channeldata[:,:,0:6]
    a_nomal = (a-a.min())/(a.max()-a.min())
    b = channeldata[:,:,6:14]
    b_nomal = (b-b.min())/(b.max()-b.min())

    fy_4[:,:,0:6] = a_nomal
    fy_4[:,:,6:14] = b_nomal
    fy_4_ = fy_4[100:1000,300:1300,:]
    label = (label[201:1101,100:1100])
    label = np.uint8(label)
    return label,fy_4_

def searchandsave(mode,hi_label_path,listla,fy_path_,size,savepath_fy,savepath_fy_label,savepath_fy_jpg):
    savepath_fy_jpg_xiao = savepath_fy_jpg+'xiao'
    if not os.path.exists(savepath_fy_jpg_xiao):
        os.makedirs(savepath_fy_jpg_xiao)
    for index in range(len(listla)): #
        #葵花八号读取数据
        #index = 2
        path_ri = os.path.join(hi_label_path,listla[index]) #'H:/fydatahimawari/Himawari\\202001'
        list_ri = os.listdir(path_ri) 
        for ri in range(29,len(list_ri)):
            path_shi = os.path.join(path_ri,list_ri[ri]) #'H:/fydatahimawari/Himawari\\202001\\01'
            list_shi = os.listdir(path_shi)
            for shi in range(len(list_shi)):
                if int(list_shi[shi])<=10:
                    path_fen = os.path.join(path_shi,list_shi[shi]) #'H:/fydatahimawari/Himawari\\202001\\01\\00'
                    list_fen = os.listdir(path_fen)
                    for fen in range(len(list_fen)):
                        namehi = list_fen[fen][:-3]
                        keyword = 'NC_H08_'
                        strnum = namehi.find(keyword)
                        date = namehi[strnum+7: strnum+15]
                        time = namehi[strnum+16: strnum+20]
                        mat_label_path = os.path.join(path_fen,list_fen[fen])
                        label = nctomatmask(mat_label_path)
                        
                        fy_path = os.path.join(fy_path_,date)
                        #风云四号读取数据
                        fy_list_ = os.listdir(fy_path)
                        for i in range(len(fy_list_)):
                            keyword = date
                            strnum = fy_list_[i].find(keyword)
                            if int(fy_list_[i][strnum+8:strnum+12])<=int(time) and int(fy_list_[i][strnum+23:strnum+27])>=int(time) and fy_list_[i].find('FDI')!=-1:
                                print(format(index+1),'-----',date,'-----FY4:',fy_list_[i][strnum+8:strnum+12],'--',fy_list_[i][strnum+23:strnum+27],'-->>>>himawari8:',time,)
                                namefy = fy_list_[i][:-4]
                                h5name = os.path.join(fy_path,fy_list_[i])
                                #数据处理：利用葵花八号数据校对风云四号数据集
                                label,warpedimage = correct(h5name,label)

                                img = warpedimage[:,:,[2,1,0]]*255
                                warpedimage_ = Image.fromarray(np.uint8(img))
                                warpedimage_.save(os.path.join(savepath_fy_jpg,'%s.jpg'%(namefy)))

                                lab= changecolor(label)
                                label_ = Image.fromarray(np.uint8(lab))
                                label_.save(os.path.join(savepath_fy_jpg,'%s_map.jpg'%(namefy)))

                                ############ 裁剪 #############
                                img_block_ = view_as_windows(warpedimage, (size, size, 14), step=size)
                                mask_block_ = view_as_windows(label, (size, size), step=size)
                                hang = img_block_.shape[0]
                                lie = img_block_.shape[1]

                                for i in range (hang):
                                    for j in range(lie):
                                        img_block = img_block_[i,j,0,:,:,:]
                                        mask_block = mask_block_[i,j,:,:]
                                        if len(np.where(mask_block==10)[0]) <= size*size/8:
                                            sio.savemat(savepath_fy+'/' + '%s_%d.mat'%(namefy,i*lie+j),{'data':img_block})
                                            sio.savemat(savepath_fy_label+'/' + '%s_%d.mat'%(namefy,i*lie+j),{'data':mask_block})
                                            lab= changecolor(mask_block)
                                            label_ = Image.fromarray(np.uint8(lab))
                                            label_.save(os.path.join(savepath_fy_jpg_xiao,'%s_%d_map.jpg'%(namefy,i*lie+j)))
                                            img = img_block[:,:,[2,1,0]]*255
                                            warpedimage_ = Image.fromarray(np.uint8(img))
                                            warpedimage_.save(os.path.join(savepath_fy_jpg_xiao,'%s_%d.jpg'%(namefy,i*lie+j)))

        print('date {} is saved',listla[index])

def creat_txtfile(output_path, file_list):
    with open(output_path, 'w') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')

def savemat(mat_path,output_path_txt_train,output_path_txt_test,output_path):
    mattain_path =output_path+'/mattrain'
    matmasktain_path =output_path+'/matmasktrain'
    mattest_path =output_path+'/mattest'
    matmasktest_path =output_path+'/matmasktest'
    if not os.path.exists(mattain_path):
        os.makedirs(mattain_path)
        os.makedirs(matmasktain_path)
        os.makedirs(mattest_path)
        os.makedirs(matmasktest_path)
    posset_train = []
    posset_val = []
    for line in open(output_path_txt_train,"r"): #设置文件对象并读取每一行文件
        posset_train.append(line[:-1])               #将每一行文件加入到list中
    for line in open(output_path_txt_test,"r"): #设置文件对象并读取每一行文件
        posset_val.append(line[:-1])               #将每一行文件加入到list中
    for j in range(len(posset_train)):
        image = sio.loadmat(os.path.join(mat_path+'/mattrain',posset_train[j]))['data']
        sio.savemat(mattain_path+'/' + '%s'%(posset_train[j]),{'data':image})
        label = sio.loadmat(os.path.join(mat_path+'/matmasktrain',posset_train[j]))['data']
        sio.savemat(matmasktain_path+'/' + '%s'%(posset_train[j]),{'data':label})
    for k in range(len(posset_val)):
        image = sio.loadmat(os.path.join(mat_path+'/mattrain',posset_val[k]))['data']
        sio.savemat(mattest_path+'/' + '%s'%(posset_val[k]),{'data':image})
        label = sio.loadmat(os.path.join(mat_path+'/matmasktrain',posset_val[k]))['data']
        sio.savemat(matmasktest_path+'/' + '%s'%(posset_val[k]),{'data':label})

def randompick(year,month_start,month,mat_path,output_path):
    mat_path_ = mat_path+'/mattrain'
    file_list = os.listdir(mat_path_)
    for i in range(month_start,month_start+month):
        k = str(i)
        other_url = k.zfill(2)
        date ='NOM_'+year+other_url
        l = [s for s in file_list if date in s]
        print(date,'----->',len(l))
        random.shuffle(l)
        posnm = 10000 #10000张图片一个训练
        posset_train = l[0:posnm]
        posnm_ = 500 #500张图片一个训练
        posset_val = l[posnm:posnm+posnm_]
        output_path_txt_train = './fysplit/v20/randompick/train.txt'
        creat_txtfile(output_path_txt_train, posset_train)
        output_path_txt_test = './fysplit/v20/randompick/test.txt'
        creat_txtfile(output_path_txt_test, posset_val)
        savemat(mat_path,output_path_txt_train,output_path_txt_test,output_path)    

if __name__ == "__main__":
    #数据集路径
    #训练
    print('-----------train------------------')
    hi_label_path = 'E:/fydatahimawari/Himawari'  #葵花八号标签
    listla = os.listdir(hi_label_path)
    fy_path_ = 'E:/fydatahimawari/FY4A'  #风云四号数据
    fy_list = os.listdir(fy_path_)

    #大图
    savepath_fy_jpg = './fysplit/v20/jpg' 
    if not os.path.exists(savepath_fy_jpg):
        os.makedirs(savepath_fy_jpg)
    #小图
    savepath_fy_train = './fysplit/v20/mattrain' #训练数据集
    savepath_fy_label_train = './fysplit/v20/matmasktrain' #训练标签
    if not os.path.exists(savepath_fy_train):
        os.makedirs(savepath_fy_train)
    if not os.path.exists(savepath_fy_label_train):
        os.makedirs(savepath_fy_label_train)
    size = 100
    mode = 'train'
    searchandsave(mode,hi_label_path,listla,fy_path_,size,savepath_fy_train,savepath_fy_label_train,savepath_fy_jpg)
    
    ###########随机划分训练集和验证集########### 2020年
    # metpath = './fysplit/v20'
    # outputpath = './fysplit/v20/randompick'
    # randompick('2020',1,10,metpath,outputpath)
