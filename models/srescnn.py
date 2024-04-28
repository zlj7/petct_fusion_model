import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.ReLU()(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)
        x_2 = self.bn1(x_2)
        # print(f'res x_1.size: {x_1.size()}')
        # print(f'res x_2.size: {x_2.size()}')
        x_concat = x_1 + x_2
        x_concat = self.bn2(x_concat)
        out = nn.ReLU()(x_concat)
        return out

class srescnn(nn.Module):
    def __init__(self, num_classes):
        super(srescnn, self).__init__()
        self.conv1 = ConvBlock(3, 8, stride=2)
        self.res1 = ResBlock(8, 8)
        self.res2 = ResBlock(8, 8)

        self.conv2 = ConvBlock(3, 8, kernel_size=1, stride=2, padding=0)

        self.conv3 = ConvBlock(3, 8, kernel_size=1, stride=4, padding=0)

        self.res3 = ResBlock(16, 8, stride=2)
        self.res4 = ResBlock(8, 8)
        self.res5 = ResBlock(16, 16, stride=2)
        self.res6 = ResBlock(16, 16)
        self.res7 = ResBlock(16, 128)
        self.res8 = ResBlock(128, 256)
        # ... add more layers as needed
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)  # assuming output is binary classification

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.res1(x_1)
        x_1 = self.res2(x_1)

        x_2 = self.conv2(x)

        x_concat_1 = torch.cat((x_1, x_2), 1)

        x_3 = self.conv3(x)

        x_concat_1 = self.res3(x_concat_1)
        x_concat_1 = self.res4(x_concat_1)

        x_concat_2 = torch.cat((x_concat_1, x_3), 1)

        x_concat_2 = self.res5(x_concat_2)
        x_concat_2 = self.res6(x_concat_2)
        x_concat_2 = self.res7(x_concat_2)
        x_concat_2 = self.res8(x_concat_2)

        # ... pass through more layers as needed
        x = nn.functional.adaptive_avg_pool2d(x_concat_2, (1, 1))
        x = x.view(x.size(0), -1)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        return x

model = srescnn(num_classes=4).cuda()
data = torch.rand(1, 3, 64, 64).cuda()
# print(model)
out = model(data)
# print(out)
