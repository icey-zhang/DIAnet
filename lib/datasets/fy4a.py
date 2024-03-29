# -*- coding: utf-8 -*-
"""
从FY-4A标称数据提取指定范围指定通道
Created on 2020/09
@author: zhangjiaqing
"""

import netCDF4 as nc
import numpy as np

from .projection import latlon2linecolumn


# 各分辨率文件包含的通道号
CONTENTS = {'0500M': ('Channel02',),
            '1000M': ('Channel01', 'Channel02', 'Channel03'),
            '2000M': tuple([f'Channel{x:02d}' for x in range(1, 8)]),
            '4000M': tuple([f'Channel{x:02d}' for x in range(1, 15)])}
# 各分辨率行列数
SIZES = {'0500M': 21984,
         '1000M': 10992,
         '2000M': 5496,
         '4000M': 2748}


class FY4A_AGRI_L1(object):
    """
    FY4A（AGRI一级数据）类
    """
    def __init__(self, l1name):
        """
        获得L1数据hdf5文件对象、记录读取状态
        """
        self.dataset = nc.Dataset(l1name, 'r')
        self.resolution = l1name[-15:-10]
        self.channels = {x: None for x in CONTENTS[self.resolution]}
        self.line_begin = self.dataset.getncattr('Begin Line Number')
        self.line_end = self.dataset.getncattr('End Line Number')
        # geo_range与line和column同步并对应
        self.geo_range = None
        self.line = None
        self.column = None

    def __del__(self):
        """
        确保关闭L1数据hdf5文件
        """
        self.dataset.close()

    def extract(self, channelname, geo_range=None):
        """
        最邻近插值提取
        line：行号
        column：列号
        channelname：要提取的通道名（如'Channel01'）
        返回字典
        暂时没有处理缺测值（异常）
        REGC超出范围未解决
        """
        NOMChannelname = 'NOM' + channelname
        CALChannelname = 'CAL' + channelname
        # 若geo_range没有指定，则读取整幅图像，不定标
        if geo_range is None:
            channel = self.dataset[NOMChannelname][()]
            self.channels[channelname] = channel
            return None
        geo_range = eval(geo_range)
        if self.geo_range != geo_range:
            self.geo_range = geo_range
            # 先乘1000取整是为了防止浮点数的精度误差累积
            lat_S, lat_N, lon_W, lon_E, step = \
                [int(1000 * x) for x in geo_range]
            lat = np.arange(lat_N, lat_S-1, -step) / 1000
            lon = np.arange(lon_W, lon_E+1, step) / 1000
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
            # 求geo_range对应的标称全圆盘行列号
            line, column = \
                latlon2linecolumn(lat_mesh, lon_mesh, self.resolution)
            self.line = np.rint(line).astype(np.int_64) - self.line_begin
            self.column = np.rint(column).astype(np.int_64)
        # DISK全圆盘数据和REGC中国区域数据区别在起始行号和终止行号
        channel = \
            self.dataset[NOMChannelname][()][self.line, self.column]
        # 定标表
        CALChannel = self.dataset[CALChannelname][()].astype(np.float32)
        if NOMChannelname != 'NOMChannel07':
            CALChannel = np.append(CALChannel, np.nan)
            channel[channel >= 65534] = 4096
        else:
            CALChannel[65535] = np.nan
        self.channels[channelname] = CALChannel[channel]


# # 演示导出指定范围数据到一个.nc文件
# if __name__ == '__main__':
#     from os import listdir
#     from os.path import join
#     from datetime import datetime
#     from netCDF4 import date2num, Dataset as ncDataset
#     from matplotlib import pyplot as plt
#     h5path = r'..\data'  # FY-4A一级数据所在路径
#     ncname = r'..\data\test.nc'
#     h5list = [join(h5path, x) for x in listdir(h5path)
#               if '4000M' in x and 'FDI' in x]
#     geo_range = '10, 54, 70, 140, 0.05'
#     lat_S, lat_N, lon_W, lon_E, step = eval(geo_range)
#     lat = np.arange(lat_N, lat_S-0.01, -step) 
#     lon = np.arange(lon_W, lon_E+0.01, step)
#     channelnames = ('Channel12',)  # 测试数据Channel02有问题
#     # 创建nc文件
#     ncfile = ncDataset(ncname, 'w', format='NETCDF4')
#     ncfile.createDimension('lat', len(lat))
#     ncfile.createDimension('lon', len(lon))
#     ncfile.createDimension('time')  # 不限长
#     nclat = ncfile.createVariable('lat', 'f4', ('lat',))
#     nclon = ncfile.createVariable('lon', 'f4', ('lon',))
#     nctime = ncfile.createVariable('time', 'f8', ('time',))
#     nctime.units = 'minutes since 0001-01-01 00:00:00.0'
#     t = 0
#     for channelname in channelnames:
#         ncfile.createVariable(channelname, 'f4', ('time', 'lat', 'lon'))
#     ncfile.set_auto_mask(False)
#     # 向nc文件中写入
#     nclat[:] = lat
#     nclon[:] = lon
#     lon, lat = np.meshgrid(lon, lat)
#     for l1name in h5list:
#         fy4a_h5 = FY4A_H5(l1name, channelnames)
#         print('FY4A_H5实例化成功')
#         for channelname in channelnames:
#             fy4a_h5.extract(channelname, geo_range)
#             ncfile[channelname][t, :, :] = fy4a_h5.channels[channelname]
#             print(channelname + '读取成功')
#         time = datetime.strptime(l1name[-45: -33], '%Y%m%d%H%M%S')
#         nctime[t] = date2num(time, nctime.units)
#         ncfile.sync()  # 手动写入硬盘
#         t += 1
#         plt.figure(l1name[-45: -31])
#         plt.imshow(fy4a_h5.channels['Channel12'], cmap='gray_r')
#         plt.show()
#     ncfile.close()
