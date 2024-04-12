# -*- coding: utf-8 -*-
from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch

ResNet_list = {'resnet50': resnet50, 'resnet34': resnet34, 'resnet18': resnet18}
res_channels = {'resnet50': [64, 256, 512, 1024, 2048],
                'resnet34': [64, 64, 128, 256, 512],
                'resnet18': [64, 64, 128, 256, 512]}

class ThreeInOne(nn.Module):
    def __init__(self, args):
        super(ThreeInOne, self).__init__()
        num_classes = args.num_classes
        self.backbone_ct_img = resnet18(pretrained=True)
        self.backbone_ct_img.conv1 = nn.Conv2d(480, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        self.backbone_pet_img = resnet18(pretrained=True)
        self.backbone_pet_img.conv1 = nn.Conv2d(336, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        self.backbone_ct = resnet18(pretrained=True)
        self.backbone_ct.conv1 = nn.Conv2d(480, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        self.backbone_pet = resnet18(pretrained=True)
        self.backbone_pet.conv1 = nn.Conv2d(336, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)


        self.base_ct_img = nn.Sequential(*list(self.backbone_ct_img.children()))
        self.feature_ct_img = [self.base_ct_img[0:4], self.base_ct_img[4],
                            self.base_ct_img[5], self.base_ct_img[6], self.base_ct_img[7]]

        self.base_pet_img = nn.Sequential(*list(self.backbone_pet_img.children()))
        self.feature_pet_img = [self.base_pet_img[0:4], self.base_pet_img[4],
                            self.base_pet_img[5], self.base_pet_img[6], self.base_pet_img[7]]

        self.base_ct = nn.Sequential(*list(self.backbone_ct.children()))
        self.feature_ct = [self.base_ct[0:4], self.base_ct[4],
                            self.base_ct[5], self.base_ct[6], self.base_ct[7]]
        
        self.base_pet = nn.Sequential(*list(self.backbone_pet.children()))
        self.feature_pet = [self.base_pet[0:4], self.base_pet[4],
                            self.base_pet[5], self.base_pet[6], self.base_pet[7]]

        self.channel_num = res_channels['resnet50']
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc_ct = nn.Linear(2000, 2000, bias=True)
        self.fc_pet = nn.Linear(2000, 2000, bias=True)
        self.fc_final = nn.Linear(4000, num_classes, bias=True)
        # 添加 dropout 层
        self.dropout = nn.Dropout(p=0.5)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, ct_data, pet_data):
        bs = ct_data.size(0)
        ct_img, ct = torch.split(ct_data, [480, 480], dim=1)
        pet_img, pet = torch.split(pet_data, [336, 336], dim=1)

        # ct_img, pet_img, ct, pet = self.tsb([ct_img, pet_img, ct, pet])
        ct_img = self.backbone_ct_img(ct_img)
        pet_img = self.backbone_pet_img(pet_img)
        ct = self.backbone_ct(ct)
        pet = self.backbone_pet(pet)

        # ct_img, pet_img, ct, pet = self.pool(ct_img), self.pool(pet_img), self.pool(ct), self.pool(pet)
        x_ct = torch.cat([ct_img,
                       ct], dim=-1)
        x_ct = self.fc_ct(x_ct)
        x_ct = self.dropout(x_ct)  # 在全连接层之后添加 dropout
        x_pet = torch.cat([pet_img,
                       pet], dim=-1)
        x_pet = self.fc_pet(x_pet)
        x_pet = self.dropout(x_pet)  # 在全连接层之后添加 dropout
        x = torch.cat([x_ct, x_pet], dim=1)
        # print(f"x_ct: {x_ct.shape}")
        # print(f"x_pet: {x_pet.shape}")
        # print(f"x: {x.shape}")
        x = self.fc_final(x)

        # loss = self.lossfn(x, tgt)

        return x #, loss


class Args():
    def __init__(self):
        self.num_classes = 2

if __name__ == '__main__':
    args = Args()
    ct_data = torch.rand(4, 960, 512, 512).cuda()
    pet_data = torch.rand(4, 672, 512, 512).cuda()
    tgt = torch.tensor([1, 1, 0, 0], dtype=torch.long).cuda()
    net = ThreeInOne(args).cuda()
    out = net(ct_data, pet_data)
    print(out)
