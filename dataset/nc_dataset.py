import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
    def __init__(self, train_ct_files, train_pet_files, train_fuse_files, train_labels_files, ct_mean=None, ct_std=None, pet_mean=None, pet_std=None, fuse_mean=None, fuse_std=None, channel_mean=None, channel_std=None, train=True):
        # 初始化空列表以存储图像和标签
        ct_list = []
        pet_list = []
        fuse_list = []
        labels_list = []

        # 遍历所有文件并加载数据i
        for ct_file, pet_file, fuse_file, lbl_file in zip(train_ct_files, train_pet_files, train_fuse_files, train_labels_files):
            print(ct_file)
            ct_list.append(np.load(f'/data3/share/Shanghai_Pulmonary/NCdata/{ct_file}', allow_pickle=True))
            print(pet_file)
            pet_list.append(np.load(f'/data3/share/Shanghai_Pulmonary/NCdata/{pet_file}', allow_pickle=True))
            print(fuse_file)
            fuse_list.append(np.load(f'/data3/share/Shanghai_Pulmonary/NCdata/{fuse_file}', allow_pickle=True))
            print(lbl_file)
            labels_list.append(np.load(f'/data3/share/Shanghai_Pulmonary/NCdata/{lbl_file}', allow_pickle=True))
            # 使用新的函数来加载和调整图像大小
            # ct_list.append(load_and_resize(f'/data3/share/Shanghai_Pulmonary/NCdata/{ct_file}'))
            # pet_list.append(load_and_resize(f'/data3/share/Shanghai_Pulmonary/NCdata/{pet_file}'))
            # fuse_list.append(load_and_resize(f'/data3/share/Shanghai_Pulmonary/NCdata/{fuse_file}'))
            # labels_list.append(load_and_resize(f'/data3/share/Shanghai_Pulmonary/NCdata/{lbl_file}'))
            
        # 将所有图像和标签堆叠在一起
        self.ct = np.concatenate(ct_list, axis=0)
        # print(self.ct.shape)
        self.pet = np.concatenate(pet_list, axis=0)
        self.fuse = np.concatenate(fuse_list, axis=0)
        self.channel_3_img = np.stack((self.ct, self.pet, self.fuse), axis=1)
        self.labels = np.concatenate(labels_list, axis=0)
        # np.save('/data3/share/Shanghai_Pulmonary/NCdata_512/train_ct.npy', self.ct)
        # np.save('/data3/share/Shanghai_Pulmonary/NCdata_512/train_pet.npy', self.pet)
        # np.save('/data3/share/Shanghai_Pulmonary/NCdata_512/train_fuse.npy', self.fuse)
        # np.save('/data3/share/Shanghai_Pulmonary/NCdata_512/train_labels.npy', self.labels)

        # 将numpy数组转换为torch张量
        self.ct = torch.from_numpy(self.ct).float()
        self.pet = torch.from_numpy(self.pet).float()
        self.fuse = torch.from_numpy(self.fuse).float()
        self.channel_3_img = torch.from_numpy(self.channel_3_img).float()
        self.labels = torch.from_numpy(self.labels).long()

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

        # print(self.ct_mean)
        self.ct = z_score_normalization(self.ct, self.ct_mean.item(), self.ct_std.item())
        self.pet = z_score_normalization(self.pet, self.pet_mean.item(), self.pet_std.item())
        self.fuse = z_score_normalization(self.fuse, self.fuse_mean.item(), self.fuse_std.item())
        self.channel_3_img = z_score_normalization(self.channel_3_img, self.channel_mean.item(), self.channel_std.item())

        
        if train:
            self.ct = transform(self.ct)
            self.pet = transform(self.pet)
            self.fuse = transform(self.fuse)
            self.channel_3_img = transform(self.channel_3_img)


    def __len__(self):
        # 返回数据集的大小
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回给定索引的图像和标签
        return {
            "ct_img": self.ct[idx],
            "pet_img": self.pet[idx],
            "fusion_img": self.fuse[idx],
            "channel_3_img": self.channel_3_img[idx],
            "label": self.labels[idx],
            "ct_mean": self.ct_mean,
            "ct_std": self.ct_std,
            "pet_mean": self.pet_mean,
            "pet_std": self.pet_std,
            "fuse_mean": self.fuse_mean,
            "fuse_std": self.fuse_std,
            "channel_mean": self.channel_mean,
            "channel_std": self.channel_std
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
    train_ct_files = [f'sh_pu_ct_train_30_fold_all.npy']
    train_pet_files = [f'sh_pu_pet_train_30_fold_all.npy']
    train_fuse_files = [f'sh_pu_fuse_train_30_fold_all.npy']
    train_labels_files = [f'sh_pu_label_train_30_fold_all.npy']

    dataset = petct_dataset(train_ct_files, train_pet_files, train_fuse_files, train_labels_files)
    print(len(dataset))
    train_load = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4,
                                                drop_last=True)

    # test_dataset = petct_dataset(test_ct_files, test_pet_files, test_fuse_files, test_labels_files)
    # print(len(test_dataset))
    # test_load = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4,
    #                                             drop_last=True)

    for i, item in enumerate(train_load):
        ct_img = item['ct_img']
        pet_img = item['pet_img']
        fusion_img = item['fusion_img']
        channel_3_img = item['channel_3_img']
        label = item['label']
        print(ct_img.shape) # torch.Size([4, 64, 64])
        print(pet_img.shape) # torch.Size([4, 64, 64])
        print(fusion_img.shape) # torch.Size([4, 64, 64])
        print(channel_3_img.shape) # torch.Size([4, 64, 64])
        print(label.shape) # torch.Size([4, 1])
        break
