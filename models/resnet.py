# -*- coding: utf-8 -*-
from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch

ResNet_list = {'resnet50': resnet50, 'resnet34': resnet34, 'resnet18': resnet18}
res_channels = {'resnet50': [64, 256, 512, 1024, 2048],
                'resnet34': [64, 64, 128, 256, 512],
                'resnet18': [64, 64, 128, 256, 512]}

class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
        num_classes = args.num_classes
        self.backbone_ct = resnet50(pretrained=True)
        self.backbone_ct.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.backbone_pet = resnet50(pretrained=True)
        self.backbone_pet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
                            
        self.backbone_fusion = resnet50(pretrained=True)
        self.backbone_fusion.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.channel_num = res_channels['resnet50']
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.fc_ct = nn.Linear(1000, 2000, bias=True)
        # self.fc_pet = nn.Linear(2000, 2000, bias=True)
        self.fc_final = nn.Linear(3000, num_classes, bias=True)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, ct_data, pet_data, fusion_data):
        bs = ct_data.size(0)
        # ct_img, ct = torch.split(ct_data, [480, 480], dim=1)
        # pet_img, pet = torch.split(pet_data, [336, 336], dim=1)

        # ct_img, pet_img, ct, pet = self.tsb([ct_img, pet_img, ct, pet])
        # ct_img = self.backbone_ct_img(ct_data)
        # pet_img = self.backbone_pet_img(pet_img)
        ct = self.backbone_ct(ct_data)
        pet = self.backbone_pet(pet_data)
        fusion = self.backbone_fusion(fusion_data)
        # pet = self.backbone_pet(pet)

        # ct_img, pet_img, ct, pet = self.pool(ct_img), self.pool(pet_img), self.pool(ct), self.pool(pet)
        # x_ct = torch.cat([ct_img,
        #                ct], dim=-1)
        x = torch.cat([ct, pet, fusion], dim=1)
        # x = torch.cat([x, fusion], dim=1)
        # x_pet = torch.cat([pet_img,
        #                pet], dim=-1)
        # x_pet = self.fc_pet(x_pet)
        # x = torch.cat([x_ct, x_pet], dim=1)
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
    ct_data = torch.rand(4, 3, 64, 64).cuda()
    pet_data = torch.rand(4, 3, 64, 64).cuda()
    fusion_data = torch.rand(4, 3, 64, 64).cuda()
    tgt = torch.tensor([1, 1, 0, 0], dtype=torch.long).cuda()
    net = Resnet(args).cuda()
    out = net(ct_data, pet_data, fusion_data)
    print(out)
