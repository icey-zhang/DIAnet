import cv2
from PIL import Image
from skimage.io import imread
from skimage.io import imsave
import numpy as np
import os 

#step1: 生成只带有cloud标签的label
# readpath = "/home/data4/zjq/l8cloudmasks/mask/"
# list_ri = os.listdir(readpath)
# savepath = "/home/data4/zjq/l8cloudmasks/label/" #只带有cloud标签的label
# for ri in list_ri:
#     label_png = imread(readpath + ri)
#     label = np.zeros(label_png.shape)
#     label[np.where(label_png==(255, 255, 255))]=255
#     lastlabel = label[:,:,0].astype(np.uint8)
#     imsave(savepath + ri.replace("mask","label"),lastlabel)

#step1: 切分数据集3:1,并且裁剪成200,间隔是200，总共一张1000*1000的图像可以裁剪成25张
# from PIL import Image
from skimage import io
from skimage.util import view_as_windows

# 定义裁剪图像和标签的函数
def crop_images(image_path, label_path, crop_size=(200, 200), step=200, save_folder=None):
    image = io.imread(image_path)
    label = io.imread(label_path)
    
    # 确保图像和标签具有相同的尺寸
    assert image.shape[:2] == label.shape[:2], "Image and label must have the same dimensions"
    
    # 计算裁剪的数量
    num_patches_y = (image.shape[0] - crop_size[1]) // step + 1
    num_patches_x = (image.shape[1] - crop_size[0]) // step + 1
    
    # 裁剪图像和标签
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # 裁剪图像和标签
            crop_img = image[y * step:y * step + crop_size[1], x * step:x * step + crop_size[0]]
            crop_label = label[y * step:y * step + crop_size[1], x * step:x * step + crop_size[0]]
            
            # 构造保存的文件名
            patch_filename = f"{y * num_patches_x + x + 1:01d}_{os.path.basename(image_path)}"
            io.imsave(os.path.join(save_folder['images'], patch_filename), crop_img)
            io.imsave(os.path.join(save_folder['labels'], patch_filename.replace('_data.tif', '_label.png')), crop_label)

# 定义保存路径
base_dir = '/home/data4/zjq/l8cloudmasks'
train_dir = {
    'images': os.path.join(base_dir, 'train200v1/images'),
    'labels': os.path.join(base_dir, 'train200v1/labels')
}
test_dir = {
    'images': os.path.join(base_dir, 'test200v1/images'),
    'labels': os.path.join(base_dir, 'test200v1/labels')
}

# 确保保存文件夹存在
for folder in [train_dir['images'], train_dir['labels'], test_dir['images'], test_dir['labels']]:
    os.makedirs(folder, exist_ok=True)

# 读取图像和标签文件列表
img_folder = os.path.join(base_dir, 'img')
label_folder = os.path.join(base_dir, 'label')
img_files = [f for f in os.listdir(img_folder) if f.endswith('_data.tif')]
label_files = [f.replace('_data.tif', '_label.png') for f in img_files]

# 随机划分数据集
# test_size = 0.25
# indices = np.arange(len(img_files))
# np.random.shuffle(indices)
# split_idx = int(test_size * len(indices))
# train_indices = indices[split_idx:]
# test_indices = indices[:split_idx]
from sklearn.model_selection import train_test_split
test_size = 0.25
test_indices = np.zeros(int(test_size * len(img_files)))
train_indices = np.zeros(int((1-test_size) * len(img_files)))
# train_indices, test_indices = train_test_split(np.arange(len(img_files)), test_size=test_size, random_state=42)
test_name = []
with open("test.txt", 'r', encoding='utf-8') as file:
    # 逐行读取
    for line in file:
        # 处理每一行
        test_name.append(line.strip())
        print(line.strip())  # 使用strip()去除可能的前后空白字符
index_test = 0
index_train = 0
for i,label_name in enumerate(label_files):
    if label_name in test_name:
        print("test{}:".format(index_test),label_name)
        test_indices[index_test] = int(i)
        index_test = index_test + 1
    else:
        print("train{}:".format(index_train),label_name)
        train_indices[index_train] = int(i)
        index_train = index_train + 1

train_indices = train_indices.astype(int)
test_indices = test_indices.astype(int)
# 裁剪图像和标签，并保存到训练集和测试集的文件夹
for idx in train_indices:
    img_file = img_files[idx]
    label_file = label_files[idx]
    img_path = os.path.join(img_folder, img_file)
    label_path = os.path.join(label_folder, label_file)
    
    # 保存到训练集文件夹
    save_folder = train_dir
    crop_images(img_path, label_path, save_folder=save_folder)

for idx in test_indices:
    img_file = img_files[idx]
    label_file = label_files[idx]
    img_path = os.path.join(img_folder, img_file)
    label_path = os.path.join(label_folder, label_file)
    
    # 保存到测试集文件夹
    save_folder = test_dir
    crop_images(img_path, label_path, save_folder=save_folder)


#类别颜色对应表 
# 0:0,0,0
# 1:0,0,128
# 2:0,0,255
# 3:0,255,255
# 4:128,128,128
# 5:255,255,255 Cloud
# 6:128,128,0
