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
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0))
])
'''
MIN_ACC = 0.6774193548
'''

# ct_mean = 55.18313217163086
# ct_std = 25.414644241333008
# pet_mean = 2.817307710647583
# pet_std = 8.828560829162598
# fuse_mean = 28.745954513549805
# fuse_std = 14.085247993469238
# channel3_mean = 28.915468215942383
# channel3_std = 27.64883804321289

class petct_dataset(data_utils.Dataset):

    def __init__(self, ct_mean, ct_std, pet_mean, pet_std, fuse_mean, fuse_std, channel_mean, channel_std, txt_path=None, dataset='petct_dataset', transform=None, alpha=0.5):
        self.data_list = []
        self.items = []
        self.transform = transform
        self.alpha = alpha
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.pet_mean = pet_mean
        self.pet_std = pet_std
        self.fuse_mean = fuse_mean
        self.fuse_std = fuse_std
        self.channel_mean = channel_mean
        self.channel_std = channel_std

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
        df = pd.read_excel('/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET临床信息-睿医.xlsx') # 读取.xlsx文件
        recist = df.loc[df['影像组学序列号'] == int(name), 'RECIST'].values[0] # 查询RECIST
        # label = 1 if recist == 'CR' or recist == 'PR' else 0 # 分配标签
        recist_to_label = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}
        label = recist_to_label.get(recist, 2)  # 如果recist的值不在字典中，返回2

        # CT数据及mask
        # ct_img = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/{name}/{name}.nii.gz")
        # ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkUInt8)
        # ct_img = sitk.GetArrayFromImage(ct_img)
        # 文件夹路径
        ct_path = f'/data3/share/Shanghai_Pulmonary/sliced_img_30/ct/{name}/'
        
        # 获取文件夹中的文件名，并按照文件名排序
        filenames = sorted(os.listdir(ct_path), key=lambda x: int(re.sub('\D', '', x)))
        ct_img = np.empty((len(filenames), 64, 64))

        # 逐个读取图像文件
        i = 0
        for filename in filenames:
            if not os.path.exists(f'/data3/share/Shanghai_Pulmonary/sliced_img_30/pet/{name}/{filename}'):
                continue
            # 完整的文件路径
            file_path = os.path.join(ct_path, filename)
            # print(file_path)
            
            # 使用OpenCV读取图像
            # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # img = img.resize((64, 64))
            # CT数据
            # 打开PNG文件
            img = Image.open(file_path)
            img = img.resize((64, 64))
            # img = np.array(img)
            # img = (img - ct_mean) / ct_std
            # img = Image.fromarray(img)
            
            img = data_transforms(img)
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
        pet_path = f'/data3/share/Shanghai_Pulmonary/sliced_img_30/pet/{name}/'
        
        # 获取文件夹中的文件名，并按照文件名排序
        filenames = sorted(os.listdir(pet_path), key=lambda x: int(re.sub('\D', '', x)))
        pet_img = np.empty((len(filenames), 64, 64))

        # 逐个读取图像文件
        cnt = 0
        for filename in filenames:
            if not os.path.exists(f'/data3/share/Shanghai_Pulmonary/sliced_img_30/ct/{name}/{filename}'):
                continue
            # 完整的文件路径
            file_path = os.path.join(pet_path, filename)
            
            # 使用OpenCV读取图像
            img = Image.open(file_path)
            
            # 将PIL图像转换为NumPy数组
            img = Image.open(file_path)
            img = img.resize((64, 64))
            # img = np.array(img)
            # img = (img - pet_mean) / pet_std
            # img = Image.fromarray(img)
            img = data_transforms(img)
            
            # 将图像添加到列表中
            pet_img[cnt] = img
            cnt += 1

        fusion_img = np.empty((cnt, 64, 64), dtype=np.uint8)
        channel_3_img = np.empty((cnt, 3, 64, 64), dtype=np.uint8)
        for i in range(cnt):
            channel_3_img[i][0] = np.array(ct_img[i])
            channel_3_img[i][1] = np.array(pet_img[i])
            fusion_img[i] = np.array(ct_img[i]) * self.alpha + np.array(pet_img[i]) * (1. - self.alpha)
            channel_3_img[i][2] = fusion_img[i]
            #print(channel_3_img[i][2])
        try:
            ct_img = torch.tensor(np.ascontiguousarray(ct_img)).float()
            pet_img = torch.tensor(np.ascontiguousarray(pet_img)).float()
            fusion_img = torch.tensor(np.ascontiguousarray(fusion_img)).float()
        except:
            print('?')
        


        return {
            "ct_img": (np.array(ct_img) - self.ct_mean.item()) / self.ct_std.item(),
            "pet_img": (np.array(pet_img) - self.pet_mean.item()) / self.pet_std.item(),
            "fusion_img": (fusion_img - self.fuse_mean.item()) / self.fuse_std.item(),
            "channel_3_img": (channel_3_img - self.channel_mean.item()) / self.channel_std.item(),
            "label": label
        } #ct_img, pet_img, label


    def __len__(self):
        return len(self.items)
        
if __name__ == '__main__':
    dataset = petct_dataset('./train_6.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    ct_list = []
    pet_list = []
    fusion_list = []
    label_list_1 = []
    for i, item in enumerate(train_load):
        channel_3_img = item['channel_3_img'].squeeze(0)
        ct_img = item['ct_img'].squeeze(0)
        pet_img = item['pet_img'].squeeze(0)
        fusion_img = item['fusion_img'].squeeze(0)
        bs = len(channel_3_img)
        label_value = item['label']
        label = torch.full((bs,), label_value.item())
        

        ct_list.append(np.array(ct_img))
        pet_list.append(np.array(pet_img))
        fusion_list.append(np.array(fusion_img))
        label_list_1.append(np.array(label))

    ct = np.concatenate(ct_list, axis=0)
    pet = np.concatenate(pet_list, axis=0)
    fusion = np.concatenate(fusion_list, axis=0)
    label_list = np.concatenate(label_list_1, axis=0)
    label_list = label_list.reshape((-1, 1))

    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_ct_train_30.npy',ct)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_pet_train_30.npy',pet)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_fuse_train_30.npy',fusion)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_label_train_30.npy',label_list)

    test_dataset = petct_dataset('./test_6.txt', 'test_dataset')
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    ct_list = []
    pet_list = []
    fusion_list = []
    label_list_1 = []
    for i, item in enumerate(test_load):
        channel_3_img = item['channel_3_img'].squeeze(0)
        ct_img = item['ct_img'].squeeze(0)
        pet_img = item['pet_img'].squeeze(0)
        fusion_img = item['fusion_img'].squeeze(0)
        bs = len(channel_3_img)
        label_value = item['label']
        label = torch.full((bs,), label_value.item())

        ct_list.append(np.array(ct_img))
        pet_list.append(np.array(pet_img))
        fusion_list.append(np.array(fusion_img))
        label_list_1.append(np.array(label))

    ct = np.concatenate(ct_list, axis=0)
    pet = np.concatenate(pet_list, axis=0)
    fusion = np.concatenate(fusion_list, axis=0)
    label_list = np.concatenate(label_list_1, axis=0)
    label_list = label_list.reshape((-1, 1))
    print(label_list.shape)
        #print(channel_3_img.shape)
        #print(label.item()) # torch.Size([1])
        # break
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_ct_test_30.npy',ct)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_pet_test_30.npy',pet)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_fuse_test_30.npy',fusion)
    # np.save('/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_label_test_30.npy',label_list)