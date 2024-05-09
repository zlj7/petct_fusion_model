import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle

def z_score_normalization(x, mean, std):
    x_normalized = (x - mean) / std
    return x_normalized

# 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0))
])

class petct_dataset(Dataset):
    def __init__(self, patient_files=None, ct_mean=None, ct_std=None, pet_mean=None, pet_std=None, fuse_mean=None, fuse_std=None, channel_mean=None, channel_std=None, train=False):
        # 初始化空列表以存储图像和标签
        

        # Load patient data
        with open(patient_files, 'rb') as f:
            self.patient_datas = pickle.load(f)
        self.patient_list = list(self.patient_datas.keys())
        print(self.patient_list)
        # # 将numpy数组转换为torch张量
        # self.ct = torch.from_numpy(self.ct).float()
        # self.pet = torch.from_numpy(self.pet).float()
        # self.fuse = torch.from_numpy(self.fuse).float()
        # self.channel_3_img = torch.from_numpy(self.channel_3_img).float()
        # self.labels = torch.from_numpy(self.labels).long()

        if train:
            self.ct_mean = torch.mean(self.ct)
            self.ct_std = torch.std(self.ct)
            self.pet_mean = torch.mean(self.pet)
            self.pet_std = torch.std(self.pet)
            self.fuse_mean = torch.mean(self.fuse)
            self.fuse_std = torch.std(self.fuse)
            self.channel_mean = torch.mean(self.channel_3_img)
            self.channel_std = torch.std(self.channel_3_img)
        else:
            self.ct_mean = ct_mean
            self.ct_std = ct_std
            self.pet_mean = pet_mean
            self.pet_std = pet_std
            self.fuse_mean = fuse_mean
            self.fuse_std = fuse_std
            self.channel_mean = channel_mean
            self.channel_std = channel_std

        # # print(self.ct_mean)
        # self.ct = z_score_normalization(self.ct, self.ct_mean.item(), self.ct_std.item())
        # self.pet = z_score_normalization(self.pet, self.pet_mean.item(), self.pet_std.item())
        # self.fuse = z_score_normalization(self.fuse, self.fuse_mean.item(), self.fuse_std.item())
        # self.channel_3_img = z_score_normalization(self.channel_3_img, self.channel_mean.item(), self.channel_std.item())

        
        if train:
            self.ct = transform(self.ct)
            self.pet = transform(self.pet)
            self.fuse = transform(self.fuse)
            self.channel_3_img = transform(self.channel_3_img)


    def __len__(self):
        # 返回数据集的大小
        return len(self.patient_datas)

    def __getitem__(self, idx):
        # 返回给定索引的图像和标签

        patient = self.patient_list[idx]
        patient_data = self.patient_datas[patient]
        # print(patient_data)

        # print(self.channel_mean)
        # print(self.channel_std)
        ct = z_score_normalization(patient_data['ct'], self.ct_mean.item(), self.ct_std.item())
        pet = z_score_normalization(patient_data['pet'], self.pet_mean.item(), self.pet_std.item())
        fuse = z_score_normalization(patient_data['fusion'], self.fuse_mean.item(), self.fuse_std.item())
        channel_3_img = z_score_normalization(patient_data['channel_3_img'], self.channel_mean.item(), self.channel_std.item())

        return {
            "patient" : patient,
            "ct": ct,
            "pet": pet,
            "fusion": fuse,
            "channel_3_img" : channel_3_img,
            "label": patient_data['label']
            # "ct_img": self.ct[idx],
            # "pet_img": self.pet[idx],
            # "fusion_img": self.fuse[idx],
            # "channel_3_img": self.channel_3_img[idx],
            # "label": self.labels[idx],
            # "ct_mean": self.ct_mean,
            # "ct_std": self.ct_std,
            # "pet_mean": self.pet_mean,
            # "pet_std": self.pet_std,
            # "fuse_mean": self.fuse_mean,
            # "fuse_std": self.fuse_std,
            # "channel_mean": self.channel_mean,
            # "channel_std": self.channel_std
        } #ct_img, pet_img, label

if __name__ == '__main__':
    # 使用文件列表初始化数据集
    # train_ct_files = ['hebeixptrainct2_64.npy', 'sphxptrainct2_64.npy','sh_pu_ct_train.npy']
    # train_pet_files = ['hebeixptrainpet2_64.npy', 'sphxptrainpet2_64.npy','sh_pu_pet_train.npy']
    # train_fuse_files = ['hebeixptrainfuse2_64.npy', 'sphxptrainfuse2_64.npy','sh_pu_fuse_train.npy']
    # train_labels_files = ['hebeiytrain.npy', 'sphytrain.npy','sh_pu_label_train.npy']

    # test_ct_files = ['hebeixptestct2_64.npy', 'sphxptestct2_64.npy','sh_pu_ct_test.npy']
    # test_pet_files = ['hebeixptestpet2_64.npy', 'sphxptestpet2_64.npy','sh_pu_pet_test.npy']
    # test_fuse_files = ['hebeixptestfuse2_64.npy', 'sphxptestfuse2_64.npy','sh_pu_fuse_test.npy']
    # test_labels_files = ['hebeiytest.npy', 'sphytest.npy','sh_pu_label_test.npy']

    # dataset = petct_dataset(train_ct_files, train_pet_files, train_fuse_files, train_labels_files)
    # print(len(dataset))
    # train_load = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4,
    #                                             drop_last=True)

    # test_dataset = petct_dataset(test_ct_files, test_pet_files, test_fuse_files, test_labels_files)
    # print(len(test_dataset))
    # test_load = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4,
    #                                             drop_last=True)

    # train_ct_files = ['hebeixptrainct2_64.npy', 'sphxptrainct2_64.npy']
    # train_pet_files = ['hebeixptrainpet2_64.npy', 'sphxptrainpet2_64.npy']
    # train_fuse_files = ['hebeixptrainfuse2_64.npy', 'sphxptrainfuse2_64.npy']
    # train_labels_files = ['hebeiytrain.npy', 'sphytrain.npy']

    # test_ct_files = ['hebeixptestct2_64.npy', 'sphxptestct2_64.npy']
    # test_pet_files = ['hebeixptestpet2_64.npy', 'sphxptestpet2_64.npy']
    # test_fuse_files = ['hebeixptestfuse2_64.npy', 'sphxptestfuse2_64.npy']
    # test_labels_files = ['hebeiytest.npy', 'sphytest.npy']
    patient_files = 'sh_pu_ct_test_30_patient_fold_1.npy'

    dataset = petct_dataset(patient_files, train=False)
    print(len(dataset))
    train_load = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                                                drop_last=True)

    # test_dataset = petct_dataset(test_ct_files, test_pet_files, test_fuse_files, test_labels_files)
    # print(len(test_dataset))
    # test_load = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4,
    #                                             drop_last=True)

    for i, item in enumerate(train_load):
        print(item['patient'])
        print(item['label'])
        # ct_img = item['ct_img']
        # pet_img = item['pet_img']
        # fusion_img = item['fusion_img']
        # channel_3_img = item['channel_3_img']
        # label = item['label']
        # print(ct_img.shape) # torch.Size([4, 64, 64])
        # print(pet_img.shape) # torch.Size([4, 64, 64])
        # print(fusion_img.shape) # torch.Size([4, 64, 64])
        # print(channel_3_img.shape) # torch.Size([4, 64, 64])
        # print(label.shape) # torch.Size([4, 1])
        # break
