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

'''
MIN_ACC = 0.6774193548
'''


class petct_dataset(data_utils.Dataset):

    def __init__(self, txt_path=None ,dataset='petct_dataset', transform=None, fold=4, press=False, phase='all'):
        self.data_list = []
        self.items = []
        self.transform = transform
        self.press = press
        # if dataset == 'train':
        #     data_num = [i for i in range(5) if i != fold]
        # elif dataset == 'valid':
        #     data_num = [fold]

        # for n in data_num:
        #     path = '/data2/chengyi/dataset/Liver_Transplant/five_fold/tmp_fivefold/fold_{}.csv'.format(n)
        #     f = open(path, 'r')
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         self.items.append(row[1:])

        # path = "/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/"
        # # 遍历指定路径下的所有文件
        # for filename in os.listdir(path):
        #     # 获取文件名（不含后缀名）
        #     name, ext = os.path.splitext(filename)
        #     if(name == "295"):
        #         continue
        #     # 将文件名添加到列表中
        #     self.items.append(name)

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
        label = 1 if recist == 'CR' or recist == 'PR' else 0 # 分配标签

        if os.path.exists(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/{name}/{name}.nii.gz"):
            # CT数据及mask
            ct_img = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/{name}/{name}.nii.gz")
            # ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkFloat32)
            ct_img = sitk.GetArrayFromImage(ct_img)
            # print(ct_img.shape)

            ct_mask = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/{name}.nrrd")
            # ct_mask = sitk.Cast(sitk.RescaleIntensity(ct_mask, outputMaximum=2), sitk.sitkFloat32)
            ct_mask = sitk.GetArrayFromImage(ct_mask)

            # PET数据及mask
            pet_img = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/PET_image_nii/{name}/{name}.nii.gz")
            # pet_img = sitk.Cast(sitk.RescaleIntensity(pet_img), sitk.sitkFloat32)
            pet_img = sitk.GetArrayFromImage(pet_img)
            # 假设 pet_img 是你的图像数组
            resized_slices = []
            for img_slice in pet_img:
                #print(f"img_slice:{img_slice.shape}")
                resized_slice = cv2.resize(img_slice, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                resized_slices.append(resized_slice)

            pet_img = np.stack(resized_slices)
            # print(pet_img.shape)

            pet_mask = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask PET/{name}.nrrd")
            # pet_mask = sitk.Cast(sitk.RescaleIntensity(pet_mask, outputMaximum=2), sitk.sitkFloat32)
            pet_mask = sitk.GetArrayFromImage(pet_mask)
            resized_slices = []
            for img_slice in pet_mask:
                #print(f"img_slice:{img_slice.shape}")
                resized_slice = cv2.resize(img_slice, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                resized_slices.append(resized_slice)

            pet_mask = np.stack(resized_slices)
        else:
            # CT数据及mask
            ct_img = sitk.ReadImage(f"/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/CT_nii/{name}/{name}.nii.gz")
            # ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkFloat32)
            ct_img = sitk.GetArrayFromImage(ct_img)
            # print(ct_img.shape)

            ct_mask = sitk.ReadImage(f"/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/CT_MASK/{name}.nrrd")
            # ct_mask = sitk.Cast(sitk.RescaleIntensity(ct_mask, outputMaximum=2), sitk.sitkFloat32)
            ct_mask = sitk.GetArrayFromImage(ct_mask)

            # PET数据及mask
            pet_img = sitk.ReadImage(f"/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET_nii/{name}/{name}.nii.gz")
            # pet_img = sitk.Cast(sitk.RescaleIntensity(pet_img), sitk.sitkFloat32)
            pet_img = sitk.GetArrayFromImage(pet_img)
            # 假设 pet_img 是你的图像数组
            resized_slices = []
            for img_slice in pet_img:
                #print(f"img_slice:{img_slice.shape}")
                resized_slice = cv2.resize(img_slice, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                resized_slices.append(resized_slice)

            pet_img = np.stack(resized_slices)
            # print(pet_img.shape)

            pet_mask = sitk.ReadImage(f"/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET_MASK/{name}.nrrd")
            # pet_mask = sitk.Cast(sitk.RescaleIntensity(pet_mask, outputMaximum=2), sitk.sitkFloat32)
            pet_mask = sitk.GetArrayFromImage(pet_mask)
            resized_slices = []
            for img_slice in pet_mask:
                #print(f"img_slice:{img_slice.shape}")
                resized_slice = cv2.resize(img_slice, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                resized_slices.append(resized_slice)

            pet_mask = np.stack(resized_slices)

        ct = ct_mask * ct_img
        ct = self.resize(ct, channel_num=480, remove=True)
        # ct = ct[np.newaxis, ...]
        # print(name)
        # print(f'pet_mask:{pet_mask.shape}')
        # print(f'pet_img:{pet_img.shape}')
        pet = pet_mask * pet_img
        pet = self.resize(pet, channel_num=336, remove=True)
        # pet = pet[np.newaxis, ...]
        ct_img = self.resize(ct_img, channel_num=480, remove=False)
        # ct_img = ct_img[np.newaxis, ...]
        pet_img = self.resize(pet_img, channel_num=336, remove=False)
        # pet_img = pet_img[np.newaxis, ...]

        # if self.press:
        #     liver = self.resize(liver, channel_num=16, hw_ratio=0.5, remove=False)
        #     img = self.resize(img, channel_num=16, hw_ratio=0.5, remove=False)
        #     portal = self.resize(portal, channel_num=16, hw_ratio=0.5, remove=False)
        '''
        --> 读区mask和dicom
        统一数据增强之后-->分割出liver和门脉 6个方向分别扩充 10个pixel之后mask掉（img * mask）
        输出三个--> 整体图，liver，门脉
        '''

        stack_ct = np.concatenate([ct_img, ct], axis=0)
        stack_pet = np.concatenate([pet_img, pet], axis=0)
        # stack_img = Image.fromarray(stack_img)
        if self.transform:
            stack_ct = self.transform(stack_ct)
            stack_pet = self.transform(stack_pet)
        try:
            ct_img = torch.tensor(np.ascontiguousarray(stack_ct)).float()
            pet_img = torch.tensor(np.ascontiguousarray(stack_pet)).float()
        except:
            print('?')

        return {
            "ct_img": ct_img,
            "pet_img": pet_img,
            "label": label
        } #ct_img, pet_img, label

    def resize(self, img, channel_num, hw_ratio=1., remove=False):
        if not remove:
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
    dataset = petct_dataset('./test.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    for i, item in enumerate(train_load):
        ct_img = item['ct_img']
        pet_img = item['pet_img']
        label = item['label']
        # print(ct_img.shape) # torch.Size([1, 960, 512, 512])
        # print(pet_img.shape) # torch.Size([1, 672, 512, 512])
        # print(label.shape) # torch.Size([1])
        # break