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

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os 
#import conda 
#os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "basemap")
import mpl_toolkits.basemap as bm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import h5py
import matplotlib.image as mpimg
def save(out,save_path,row,col):
    import numpy.ma as ma
    mx = ma.array(out,dtype = 'float32',fill_value= 1e20)
    ######创建HDF文件######
    f_w = nc.Dataset('a.HDF','w',format = 'NETCDF4')
    f_w.createDimension('phony_dim_0',size=row)
    f_w.createDimension('phony_dim_1',size=col)
    f_w.createVariable('FY4CLT',np.float32,('phony_dim_0','phony_dim_1'))
    f_w.variables['FY4CLT'][:] = mx[:]
    f_w.setncattr('Resolution',0.05)
    f_w.setncattr('IniLat',5)
    f_w.setncattr('EndLat',55)
    f_w.setncattr('IniLon',70)
    f_w.setncattr('EndLon',140)
    f_w.close()

def save_hdf(out,save_path,row,col):
    import numpy.ma as ma
    mx = ma.array(out,dtype = 'float32',fill_value= 1e20)
    ######创建HDF文件######
    f_w = h5py.File(save_path,'w')
    f_w.create_dataset('FY4CLT',data=mx,dtype=np.float32)
    f_w.attrs['Resolution'] = float("{0:.8f}".format(0.05))
    f_w.attrs['IniLat'] = float("{0:.8f}".format(5.0))
    f_w.attrs['EndLat'] = float("{0:.8f}".format(55.0))
    f_w.attrs['IniLon'] = float("{0:.8f}".format(70.0))
    f_w.attrs['EndLon'] = float("{0:.8f}".format(140.0))
    f_w.close()


def RGB_to_Hex(tmp):
    rgb = tmp.split(' ')#将RGB格式划分开来
    strs = '#'
    for i in rgb:
        num = int(i)#将str转int
        #将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x','0').upper()
    return strs

###############color#########
def new_cmap(rgbpath,idx=None):
    f=open(rgbpath,'r')
    data=f.readlines()
    f.close()
    color = []
    for i in range(0,len(data)):
        if data[i] != '\n':
            color.append(RGB_to_Hex(data[i]))

    if idx != None:
        colors = []
        for i in idx:
            colors.append(color[i])
    else:
        colors = color
    
    cmap1 = col.ListedColormap(colors,"indexed")
    plt.cm.register_cmap(cmap=cmap1)
    return cmap1


def plotscatter_Asia(data,filename,cmap,ticks=None,vmin=None,vmax=None,bounds=None, root_address=''):
    # 输入图像为11分类结果 2020-10-10

    #plt.rcParams['font.family'] = 'Times New Roman' #字体
    #plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(7,4))
    grid = plt.GridSpec(7, 4, wspace=2.5, hspace=1)
    ax = plt.subplot(grid[0:7,0:4])
    ax = plt.subplot(121)#ax = plt.subplot(111)
    # plt.title('Time:0810_0600')
    #plt.subplots_adjust(left=0.08,right=0.88,top=0.95,bottom=0.1)
    mapobj = bm.Basemap(llcrnrlat=5, llcrnrlon=70, urcrnrlat=55,urcrnrlon=140) #projection='cyl',resolution='l',
    mapobj.drawcoastlines(linewidth=0.5,color='k')
    mapobj.drawcountries(linewidth=0.5,color='k')
    # xmajorLocator = MultipleLocator(20)
    # xminorLocator = MultipleLocator(10)
    # ymajorLocator = MultipleLocator(10)
    # yminorLocator = MultipleLocator(40)
    # ax.xaxis.set_major_locator(xmajorLocator)
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)
    plt.xticks(np.arange(70,140,10),['70°E','80°E','90°E','100°E','110°E','120°E','130°E'],fontname='STIXGeneral',fontsize=10) #
    plt.yticks(np.arange(5,55,10),['5°N','15°N','25°N','35°N','45°N'],fontname='STIXGeneral',fontsize=10) #
    plt.grid(linestyle='-.')
    norm = col.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))

    ###加路径
    mapobj.readshapefile(root_address + "/lib/IO/shapefiles/China_province","china",drawbounds=True)
    #mapobj.readshapefile("D:\\tzb云分类代码\\ensemble_model\\lib\\IO/shapefiles/China_province","china",drawbounds=True)
    #mapobj.readshapefile("./shapefiles/China_province","china",drawbounds=True)
    mapobj.readshapefile(root_address + "/lib/IO/shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    #mapobj.readshapefile("D:\\tzb云分类代码\\ensemble_model\\lib\\IO/shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    ###加路径
    #mapobj.readshapefile("./shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    #lat = (5,55)
    #lon = (70,140)
    #x,y = mapobj(lat,lon)
    #cs = mapobj.scatter(x,y,latlon=True,s=5,c=data,marker = '.',cmap=cmap,norm=norm,edgecolors='none') #创建颜色条
    data = np.flip(data,0)
    cs = mapobj.imshow(data,cmap=cmap,norm = norm)
    # position=fig.add_axes([0.85, 0.1, 0.015, 0.85])#位置[left, bottom, width, height]
    # cb=fig.colorbar(cs,cax=position,orientation='vertical')#方向 颜色条填色
    # cb.set_ticks(np.array(bounds)+0.8)
    # cb.set_ticklabels(ticks)
    
    ##添加legend
    ax = plt.subplot(grid[0:7,2:3])
    legend = mpimg.imread('legend.png')
    #print(legend.shape)
    plt.imshow(legend)
    plt.axis('off')
    #plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=0.2,hspace=0.2)
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename,dpi=1000)

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # img = cv2.imread(filename, 1)
    print(type(img))
    dst = img[0:5500, :]   # 裁剪坐标为[y0:y1, x0:x1]
    #cv2.imshow('image',dst)
    cv2.imwrite(filename,dst)
    """
    im = plt.imread(filename)
    plt.figure()
    def plti(im, **kwargs):

        plt.imshow(im, interpolation="none", **kwargs)
        plt.axis('off') # 去掉坐标轴
        #plt.show() # 弹窗显示图像
    im = im[0:5500,:,:]  # 直接切片对图像进行裁剪
    plti(im)
    plt.savefig(filename,dpi=1000)
    #png2tif(filename)
    """


def plotscatter_Asia_(data,filename,cmap,ticks=None,vmin=None,vmax=None,bounds=None, root_address=''):
    # 输入图像为11分类结果 2020-10-10

    #plt.rcParams['font.family'] = 'Times New Roman' #字体
    #plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot(121)#ax = plt.subplot(111)
    # plt.title('Time:0810_0600')
    plt.subplots_adjust(left=0.08,right=0.88,top=0.95,bottom=0.1)
    mapobj = bm.Basemap(llcrnrlat=5, llcrnrlon=70, urcrnrlat=55,urcrnrlon=140) #projection='cyl',resolution='l',
    mapobj.drawcoastlines(linewidth=0.5,color='k')
    mapobj.drawcountries(linewidth=0.5,color='k')
    xmajorLocator = MultipleLocator(20)
    xminorLocator = MultipleLocator(10)
    ymajorLocator = MultipleLocator(10)
    yminorLocator = MultipleLocator(40)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.xticks(np.arange(70,140,10),['70°E','80°E','90°E','100°E','110°E','120°E','130°E'],fontname='STIXGeneral',fontsize=10) #
    plt.yticks(np.arange(5,55,10),['5°N','15°N','25°N','35°N','45°N'],fontname='STIXGeneral',fontsize=10) #
    plt.grid(linestyle='-.')
    norm = col.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))

    ###加路径
    mapobj.readshapefile(root_address + "/lib/IO/shapefiles/China_province","china",drawbounds=True)
    #mapobj.readshapefile("D:\\tzb云分类代码\\ensemble_model\\lib\\IO/shapefiles/China_province","china",drawbounds=True)
    #mapobj.readshapefile("./shapefiles/China_province","china",drawbounds=True)
    mapobj.readshapefile(root_address + "/lib/IO/shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    #mapobj.readshapefile("D:\\tzb云分类代码\\ensemble_model\\lib\\IO/shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    ###加路径
    #mapobj.readshapefile("./shapefiles/china_nine_dotted_line","nine_dotted",drawbounds=True)
    #lat = (5,55)
    #lon = (70,140)
    #x,y = mapobj(lat,lon)
    #cs = mapobj.scatter(x,y,latlon=True,s=5,c=data,marker = '.',cmap=cmap,norm=norm,edgecolors='none') #创建颜色条
    data = np.flip(data,0)
    cs = mapobj.imshow(data,cmap=cmap,norm = norm)
    position=fig.add_axes([0.85, 0.1, 0.015, 0.85])#位置[left, bottom, width, height]
    cb=fig.colorbar(cs,cax=position,orientation='vertical')#方向 颜色条填色
    cb.set_ticks(np.array(bounds)+0.8)
    cb.set_ticklabels(ticks)
    
    ##添加legend
    ax = plt.subplot(122)
    legend = mpimg.imread('legend.png')
    plt.imshow(legend)
    plt.axis('off')
    #plt.show()
    plt.savefig(filename,dpi=1000)
    #png2tif(filename)

def png2tif(openpath):
    img = cv2.imread(openpath)
    print(openpath.replace(".png",".tif"))
    newfile = openpath.replace(".png",".tif")
    cv2.imwrite(newfile)

def scatter_Asian(mask,save_path_tif, name, root_address=''):
    # 输入图像为11分类结果 2020-10-10
    # 将第10类赋值为第0类
    mask[mask==0]=10
    mask = mask - 1

    ###加路径
    #rgbpath='D:\\tzb云分类代码\\ensemble_model\\lib\IO/colors/Cat12.rgb'
    rgbpath = root_address + '/lib/IO/colors/Cat12.rgb'
    ###加路径
    #save_path_tif = os.path.join(result_path ,'TIF',name+'_Img.tif')
    save_path_tif = os.path.join(save_path_tif, name+'_Img.tif')#'_Img.png'
    cltidx = [0,1,2,3,4,5,6,7,8,9]

    #cltidx = [1,2,3,4,5,6,7,8,9,0,10]
    cmap1 = new_cmap(rgbpath,cltidx)
    ticks = ['Ci', 'Cs', 'Dc', 'Ac','As', 'Ns', 'Cu', 'Sc', 'St', 'Fill']
    #ticks = ['无云','卷云*', '卷层云', '深对流云*', '高积云','高层云', '雨层云*', '积云*', '层积云', '层云*', '未知']
    #mask = np.argmax(mask,axis=1).squeeze()
    plotscatter_Asia(mask,save_path_tif,cmap1,ticks=ticks,vmin=0,vmax=12,bounds=np.arange(11),root_address=root_address)

if __name__ == "__main__":
    ##########数据集路径####
    #fy_path = 'E:/FY4A/fy_imagetest'  #风云四号数据 HDF 格式
    ##########加载数据#####
    #dataload(fy_path)
    result_path = '/home/amax/mnt1/ensemble_HR_and_HR_OCR/hrnet_ocr_nc_results2/prob'
    save_path_mat = os.path.join(result_path,)
    assert os.path.exists(save_path_mat)
    index = os.listdir(save_path_mat)
    mask = sio.loadmat(os.path.join(save_path_mat,index[0]))['data']
    mask = np.argmax(mask,axis=2)
    ######网络测试#########
    #mask = np.random.randint(0,10,(1000,1400))
    name = index[0][:-4]
    save_path_HDF = os.path.join(result_path ,'HDF',name+'_CLT.HDF')
    save_hdf(mask,save_path_HDF,1000,1400)
    #print(mx)
    ##############验证正确性###########
    #nc_obj = nc.Dataset('a.HDF')
    #type = nc_obj.variables.keys()
    #############tif文件###########
    rgbpath='/home/amax/mnt1/ensemble_HR_and_HR_OCR/lib/IO/colors/Cat12.rgb'
    save_path_tif = os.path.join(result_path ,'TIF',name+'_Img.tif')
    cltidx = [0,1,2,3,4,5,6,7,8,9,10]
    cmap1 = new_cmap(rgbpath,cltidx)
    ticks = ['Clear','Ci', 'Cs', 'Dc', 'Ac','As', 'Ns', 'Cu', 'Sc', 'St', 'Uncertain']
    #ticks = ['无云','卷云*', '卷层云', '深对流云*', '高积云','高层云', '雨层云*', '积云*', '层积云', '层云*', '未知']
    plotscatter_Asia(mask,save_path_tif,cmap1,ticks=ticks,vmin=0,vmax=12,bounds=np.arange(12))



