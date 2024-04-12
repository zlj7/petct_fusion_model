import random
import pandas as pd
import csv
import SimpleITK as sitk
import scipy.ndimage
import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
#from xpinyin import Pinyin
import copy
import re
import skimage
from torchvision import transforms

# 定义数据增强
data_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(64),  # 随机裁剪并缩放到 64x64
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomVerticalFlip(),  # 随机垂直翻转
    # transforms.RandomRotation(30),  # 在 (-30, 30) 范围内随机旋转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5], [0.5])  # 归一化
])

'''
MIN_ACC = 0.6774193548
'''


class petct_dataset(data_utils.Dataset):

    def __init__(self, txt_path=None ,dataset='petct_dataset', transform=None, fold=4, press=False, alpha=0.5, phase='all'):
        self.data_list = []
        self.items = []
        self.transform = transform
        self.press = press
        self.alpha = alpha

        # 打开文件
        with open(txt_path, "r") as file:
            # 逐行读取内容
            lines = file.readlines()

            # 遍历每行内容
            for line in lines:
                # 将文件名添加到列表中
                self.items.append(line.rstrip())

        print('dataset=', dataset, '\nlen=', len(self.items))

    def __getitem__(self, idx):
        name = self.items[idx]
        # 根据序列号，从xlsx中读取RECIST，进而获取分类标签
        df = pd.read_excel('/data2/zhenglujie/Shanghai_Pulmonary/PET_dataset.xlsx') # 读取.xlsx文件
        recist = df.loc[df['影像组学序列号'] == int(name), 'RECIST'].values[0] # 查询RECIST
        label = 1 if recist == 'CR' or recist == 'PR' else 0 # 分配标签

        # CT数据及mask
        # ct_img = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/{name}/{name}.nii.gz")
        # ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkUInt8)
        # ct_img = sitk.GetArrayFromImage(ct_img)
        # 文件夹路径
        folder_path = f'/data3/share/Shanghai_Pulmonary/sliced_img/ct/{name}/'
        
        # 获取文件夹中的文件名，并按照文件名排序
        filenames = sorted(os.listdir(folder_path), key=lambda x: int(re.sub('\D', '', x)))
        ct_img = np.empty((len(filenames), 64, 64))

        # 逐个读取图像文件
        i = 0
        for filename in filenames:
            
            # 完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # print(file_path)
            
            # 使用OpenCV读取图像
            # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # img = img.resize((64, 64))
            # CT数据
            # 打开PNG文件
            img = Image.open(file_path)
            img = img.resize((64, 64))
            # img = np.array(img)
            #img = data_transforms(img)
            # print(img)
            # print(img.shape)
            
            # 将图像添加到列表中
            ct_img[i] = img
            i += 1

        # print(ct_img)
        # 检查图像中是否存在 NaN 数据
        # if np.isnan(ct_img).any():
        #     print("图像中存在 NaN 数据")
        # else:
        #     print("图像中不存在 NaN 数据")
        # print(ct_img)

        # 文件夹路径
        folder_path = f'/data3/share/Shanghai_Pulmonary/sliced_img/pet/{name}/'
        
        # 获取文件夹中的文件名，并按照文件名排序
        filenames = sorted(os.listdir(folder_path), key=lambda x: int(re.sub('\D', '', x)))
        pet_img = np.empty((len(filenames), 64, 64))

        # 逐个读取图像文件
        i = 0
        for filename in filenames:
            # 完整的文件路径
            file_path = os.path.join(folder_path, filename)
            
            # 使用OpenCV读取图像
            img = Image.open(file_path)
            img = img.resize((64, 64))
            # img = np.array(img)
            #img = data_transforms(img)
            
            # 将图像添加到列表中
            pet_img[i] = img
            i += 1

        # print(pet_img)
        # ct_img = self.resize(ct_img, channel_num=64, remove=False)
        # pet_img = self.resize(pet_img, channel_num=64, remove=False)
        new_space = [64, 64, 64]
        ct_img = skimage.transform.resize(ct_img,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)
        pet_img = skimage.transform.resize(pet_img,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

        fusion_img = np.empty((64, 64, 64), dtype=np.uint8)
        for i in range(64):
            fusion_img[i] = ct_img[i] * self.alpha + pet_img[i] * (1. - self.alpha)

        try:
            ct_img = torch.tensor(np.ascontiguousarray(ct_img)).float()
            pet_img = torch.tensor(np.ascontiguousarray(pet_img)).float()
            fusion_img = torch.tensor(np.ascontiguousarray(fusion_img)).float()
        except:
            print('?')

        return {
            "ct_img": ct_img,
            "pet_img": pet_img,
            "fusion_img": fusion_img,
            "label": label
        } #ct_img, pet_img, label

    def resize(self, img, channel_num, hw_ratio=1., remove=False):
        if not remove:
            # print(img.shape)
            c, h, w = img.shape
            return scipy.ndimage.zoom(img, (channel_num / c, hw_ratio, hw_ratio), output=None, order=3, mode='nearest')
            # return cv2.resize(img, (channel_num, 512))
        else:
            c, h, w = img.shape
            if c < channel_num:
                diff0 = (channel_num - c) // 2
                diff1 = channel_num - c - diff0
                return np.concatenate([np.zeros([diff0, h, w], dtype=img.dtype),
                                       img,
                                       np.zeros([diff1, h, w], dtype=img.dtype)], axis=0)

            img_idx = img.sum(axis=1).sum(axis=1)
            img_idx_nonzero = np.argwhere(img_idx != 0)
            if len(img_idx_nonzero) == 0:
                return np.zeros([channel_num, h, w], dtype=img.dtype)
            min_channel, max_channel = img_idx_nonzero[0] - 1, img_idx_nonzero[-1] + 1
            rest = channel_num - (max_channel - min_channel + 1)
            if rest < 0:
                return scipy.ndimage.zoom(img, (channel_num / c, 1, 1), output=None, order=3, mode='nearest')
                # print(channel_num)
                # print(rest)
                # print('channel_num is set too low')
                # raise ValueError

            min_channel = min_channel - rest // 2
            max_channel = min_channel + channel_num
            if min_channel < 0:
                min_channel = 0
                max_channel = channel_num
            elif max_channel >= c:
                max_channel = c
                min_channel = c - channel_num
            img = img[int(min_channel): int(max_channel)]

            return img

    def __len__(self):
        return len(self.items)
        
if __name__ == '__main__':
    dataset = petct_dataset('./train.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    for i, item in enumerate(train_load):
        ct_img = item['ct_img']
        pet_img = item['pet_img']
        fusion_img = item['fusion_img']
        label = item['label']
        print(ct_img.shape) # torch.Size([1, 128, 512, 512])
        print(pet_img.shape) # torch.Size([1, 128, 512, 512])
        print(fusion_img.shape)
        print(label.shape) # torch.Size([1])
        break