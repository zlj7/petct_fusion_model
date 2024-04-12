# -*- coding: utf-8 -*-
from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch

ResNet_list = {'resnet50': resnet50, 'resnet34': resnet34, 'resnet18': resnet18}
res_channels = {'resnet50': [64, 256, 512, 1024, 2048],
                'resnet34': [64, 64, 128, 256, 512],
                'resnet18': [64, 64, 128, 256, 512]}

class Resnet_orin(nn.Module):
    def __init__(self, args):
        super(Resnet_orin, self).__init__()
        num_classes = args.num_classes
        self.backbone_ct = resnet18(pretrained=True)
        self.backbone_ct.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.backbone_ct.fc = nn.Linear(self.backbone_ct.fc.in_features, 2)
        self.channel_num = res_channels['resnet18']
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc_final = nn.Linear(1000, num_classes, bias=True)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, channel_3_img):
        x = self.backbone_ct(channel_3_img)
        # x = self.fc_final(x)

        return x #, loss

class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
        num_classes = args.num_classes
        self.backbone_ct = resnet18(pretrained=True)
        self.backbone_ct.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.channel_num = res_channels['resnet18']
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ���� dropout ��
        self.dropout = nn.Dropout(p=0.5)

        # ���Ӷ����ȫ���Ӳ�
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc_final = nn.Linear(250, num_classes, bias=True)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, channel_3_img):
        x = self.backbone_ct(channel_3_img)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # ��ȫ���Ӳ�֮������ dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # ��ȫ���Ӳ�֮������ dropout
        x = self.fc_final(x)
        x = torch.sigmoid(x)

        return x #, loss


class Args():
    def __init__(self):
        self.num_classes = 2

if __name__ == '__main__':
    args = Args()
    # ct_data = torch.rand(4, 128, 64, 64).cuda()
    # pet_data = torch.rand(4, 128, 512, 512).cuda()
    # fusion_data = torch.rand(4, 128, 512, 512).cuda()
    channel_3_img = torch.rand(4, 3, 64, 64).cuda()
    # tgt = torch.tensor([4, 1, 0, 0], dtype=torch.long).cuda()
    net = Resnet_orin(args).cuda()
    out = net(channel_3_img)
    print(out)
