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
from PIL import Image

'''
MIN_ACC = 0.6774193548
'''


class petct_dataset(data_utils.Dataset):

    def __init__(self, txt_path=None ,dataset='img_dataset', transform=None, alpha=0.5, fold=4, press=False, phase='all'):
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
        path = self.items[idx]
        name = path.split("/")[-2]
        nomber = path.split("/")[-1]
        # print(f"name:{name}")
        # 根据序列号，从xlsx中读取RECIST，进而获取分类标签
        df = pd.read_excel('/data2/zhenglujie/Shanghai_Pulmonary/PET_dataset.xlsx') # 读取.xlsx文件
        recist = df.loc[df['影像组学序列号'] == int(name), 'RECIST'].values[0] # 查询RECIST
        label = 1 if recist == 'CR' or recist == 'PR' else 0 # 分配标签

        # CT数据
        # 打开PNG文件
        ct = Image.open(path.replace("pet", "ct"))
        # print(f"ct:{ct.size}")
        ct = ct.resize((64, 64))
        ct = np.array(ct)
        # print(f"ct:{ct.size}")


        # PET
        pet = Image.open(path)
        pet = pet.resize((64, 64))
        pet = np.array(pet)

        fusion = ct * self.alpha + ct * (1. - self.alpha)

        fusion_img = np.empty((3, 64, 64), dtype=np.uint8)
        fusion_img[0] = ct
        fusion_img[1] = pet
        fusion_img[2] = fusion


        # '''
        # --> 读区mask和dicom
        # 统一数据增强之后-->分割出liver和门脉 6个方向分别扩充 10个pixel之后mask掉（img * mask）
        # 输出三个--> 整体图，liver，门脉
        # '''

        # stack_ct = np.concatenate([ct_img, ct], axis=0)
        # stack_pet = np.concatenate([pet_img, pet], axis=0)
        # # stack_img = Image.fromarray(stack_img)
        # if self.transform:
        #     stack_ct = self.transform(stack_ct)
        #     stack_pet = self.transform(stack_pet)
        # try:
        #     ct_img = torch.tensor(np.ascontiguousarray(stack_ct)).float()
        #     pet_img = torch.tensor(np.ascontiguousarray(stack_pet)).float()
        # except:
        #     print('?')

        return {
            "fusion_img": fusion_img.astype(float),
            "label": label
        } #ct_img, pet_img, label

    # def resize(self, img, channel_num, hw_ratio=1., remove=False):
    #     if not remove:
    #         c, h, w = img.shape
    #         return scipy.ndimage.zoom(img, (channel_num / c, hw_ratio, hw_ratio), output=None, order=3, mode='nearest')
    #         # return cv2.resize(img, (channel_num, 512))
    #     else:
    #         c, h, w = img.shape
    #         if c < channel_num:
    #             diff0 = (channel_num - c) // 2
    #             diff1 = channel_num - c - diff0
    #             return np.concatenate([np.zeros([diff0, h, w], dtype=img.dtype),
    #                                    img,
    #                                    np.zeros([diff1, h, w], dtype=img.dtype)], axis=0)

    #         img_idx = img.sum(axis=1).sum(axis=1)
    #         img_idx_nonzero = np.argwhere(img_idx != 0)
    #         if len(img_idx_nonzero) == 0:
    #             return np.zeros([channel_num, h, w], dtype=img.dtype)
    #         min_channel, max_channel = img_idx_nonzero[0] - 1, img_idx_nonzero[-1] + 1
    #         rest = channel_num - (max_channel - min_channel + 1)
    #         if rest < 0:
    #             return scipy.ndimage.zoom(img, (channel_num / c, 1, 1), output=None, order=3, mode='nearest')
    #             # print(channel_num)
    #             # print(rest)
    #             # print('channel_num is set too low')
    #             # raise ValueError

    #         min_channel = min_channel - rest // 2
    #         max_channel = min_channel + channel_num
    #         if min_channel < 0:
    #             min_channel = 0
    #             max_channel = channel_num
    #         elif max_channel >= c:
    #             max_channel = c
    #             min_channel = c - channel_num
    #         img = img[int(min_channel): int(max_channel)]

    #         return img

    def __len__(self):
        return len(self.items)
        
if __name__ == '__main__':
    dataset = petct_dataset('./train.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    for i, item in enumerate(train_load):
        fusion_img = item['fusion_img']
        label = item['label']
        print(fusion_img)
        print(label)
        # print(ct_img.shape) # torch.Size([1, 960, 512, 512])
        # print(pet_img.shape) # torch.Size([1, 672, 512, 512])
        # print(label.shape) # torch.Size([1])
        # break