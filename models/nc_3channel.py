from torchvision.models import resnet50, resnet34, resnet18
import torch.nn as nn
import torch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_SE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, y_pool, y):
        B, N, C = y.shape
        q = self.q(y_pool).reshape(B, 1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # x: (B, N, C)
        return x


class MCA(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MCA, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_SE(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, yq, ykv):
        # y = bs, h*w, c
        # y_pool = bs, 1 ,c

        y = self.drop_path(self.attn(self.norm1(yq), self.norm1(ykv)))

        return y


class spatial_block(nn.Module):
    def __init__(self, channel):
        super(spatial_block, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.MCA = MCA(dim=channel)

    def forward(self, x_q, y_kv):
        bs, c, h, w = y_kv.shape
        pool_q = self.pool(x_q)  # bs * c * 1 * 1
        pool_q = pool_q.reshape(bs, 1, c)
        y_kv = y_kv.permute(0, 2, 3, 1).reshape(bs, h * w, c)
        y_kv = self.MCA(pool_q, y_kv) * y_kv
        out = y_kv.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        return out


class channel_block(nn.Module):
    def __init__(self, channel, spatial_dim, sample=1):
        super(channel_block, self).__init__()
        self.sample = sample
        if sample != 1:
            self.pool = nn.AvgPool2d(kernel_size=(sample, sample))
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(1, 1))
        self.MCA = MCA(dim=spatial_dim, num_heads=8)

    def forward(self, x_q, y_kv):

        bs, c, h, w = y_kv.shape
        pool_q = self.conv1(x_q)  # bs * 1 * h * w
        pool_q = pool_q.reshape(bs, 1, h * w)
        y_kv = y_kv.reshape(bs, c, h * w)
        out = self.MCA(pool_q, y_kv) * y_kv

        # out = weight * y_kv
        out = out.reshape(bs, c, h, w)
        return out


class TRANS_SE_BLOCK(nn.Module):
    def __init__(self, channel_num, spatial_dim, channel_out=4):
        super(TRANS_SE_BLOCK, self).__init__()
        '''
        因为主要的目的还是对模型进行spatial的增强：不同的聚焦位置，因此channel attention的重要性和可解释性不怎么重要
        所以改为自增强+cross增强
        x代表fused
        y代表focal plane
        '''


        self.self = spatial_block(channel_num)
        self.cross = channel_block(channel_num, spatial_dim)



        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=channel_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
        )

    def forward(self, x, y):


        y = self.self(y, y)
        y = self.cross(x, y)

        y_squeeze = self.squeeze(y)
        return y, y_squeeze


class tsb(nn.Module):
    def __init__(self, channel_num, spatial_dim, channel_out=4, core='img'):
        super(tsb, self).__init__()
        self.se1 = TRANS_SE_BLOCK(channel_num=channel_num, channel_out=channel_out, spatial_dim=spatial_dim)
        self.se2 = TRANS_SE_BLOCK(channel_num=channel_num, channel_out=channel_out, spatial_dim=spatial_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel_num + 2 * channel_out, out_channels=channel_num, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(channel_num),
            nn.ReLU())

        self.core = core

    def forward(self, img, liver, portal):
        # x.shape = yi.shape = bs*N*H*W
        if self.core == 'img':
            liver, liver_squeeze = self.se1(img, liver)
            portal, portal_squeeze = self.se2(img, portal)
            img = torch.cat([img, liver_squeeze, portal_squeeze], dim=1)
            img = self.conv(img)
        elif self.core == 'liver':
            img, img_squeeze = self.se1(liver, img)
            portal, portal_squeeze = self.se2(liver, portal)
            liver = torch.cat([liver, img_squeeze, portal_squeeze], dim=1)
            liver = self.conv(liver)
        else:
            img, img_squeeze = self.se1(portal, img)
            liver, liver_squeeze = self.se2(portal, liver)
            portal = torch.cat([portal, img_squeeze, liver_squeeze], dim=1)
            portal = self.conv(portal)
        return img, liver, portal


class tsb_layer(nn.Module):
    def __init__(self, f_img, f_liver, f_portal, current_layer, fuse_layer, channel_num, spatial_dim, channel_out=4, core='img'):
        super(tsb_layer, self).__init__()
        self.featrue_img = f_img
        self.featrue_liver = f_liver
        self.featrue_portal = f_portal

        self.apply_tsb = current_layer >= fuse_layer
        if self.apply_tsb:
            self.layer = tsb(channel_num, spatial_dim, channel_out, core)

    def forward(self, input):
        img, liver, portal = input
        img = self.featrue_img(img)
        liver = self.featrue_liver(liver)
        portal = self.featrue_portal(portal)
        if self.apply_tsb:
            img, liver, portal = self.layer(img, liver, portal)
        return img, liver, portal


ResNet_list = {'resnet50': resnet50, 'resnet34': resnet34, 'resnet18': resnet18}
res_channels = {'resnet50': [64, 256, 512, 1024, 2048],
                'resnet34': [64, 64, 128, 256, 512],
                'resnet18': [64, 64, 128, 256, 512]}


class pcfnet(nn.Module):
    def __init__(self, args):
        super(pcfnet, self).__init__()
        num_classes = args.num_classes
        channel_out = 4
        fuse_layer = 3
        core = 'img'
        print('Core Component is {}'.format(core))


        self.backbone_img = resnet50(pretrained=True)
        self.backbone_img.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))
        self.backbone_liver = resnet50(pretrained=True)
        self.backbone_liver.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))
        self.backbone_portal = resnet50(pretrained=True)
        self.backbone_portal.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))


        self.base_img = nn.Sequential(*list(self.backbone_img.children()))
        self.feature_img = [self.base_img[0:4], self.base_img[4],
                            self.base_img[5], self.base_img[6], self.base_img[7]]

        self.base_liver = nn.Sequential(*list(self.backbone_liver.children()))
        self.feature_liver = [self.base_liver[0:4], self.base_liver[4],
                              self.base_liver[5], self.base_liver[6], self.base_liver[7]]

        self.base_portal = nn.Sequential(*list(self.backbone_portal.children()))
        self.feature_portal = [self.base_portal[0:4], self.base_portal[4],
                               self.base_portal[5], self.base_portal[6], self.base_portal[7]]


        self.channel_num = res_channels['resnet50']
        self.spatial_dim = [256 ** 2, 128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2]
        self.tsb = nn.Sequential(*[
            tsb_layer(self.feature_img[i], self.feature_liver[i], self.feature_portal[i], i,
                      fuse_layer, self.channel_num[i], self.spatial_dim[i], channel_out, core)
            for i in range(len(self.channel_num))])


        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(res_channels['resnet50'][-1]*3, num_classes, bias=True)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def model_name(self):
        return self.name


    def forward(self, x):
        bs = x.size(0)
        portal, liver, img = torch.split(x, [1, 1, 1], dim=1)

        img, liver, portal = self.tsb([img, liver, portal])

        img, liver, portal = self.pool(img), self.pool(liver), self.pool(portal)
        x = torch.cat([img.squeeze(dim=2).squeeze(dim=2),
                       liver.squeeze(dim=2).squeeze(dim=2),
                       portal.squeeze(dim=2).squeeze(dim=2)], dim=-1)
        x = self.fc(x)
        # loss = self.lossfn(x, tgt)

        return x#, loss

class Args():
    def __init__(self):
        self.num_classes = 2


if __name__ == '__main__':
    args = Args()
    data = torch.rand(4, 3, 64, 64).cuda()
    tgt = torch.tensor([1, 1, 0, 0], dtype=torch.long).cuda()
    net = pcfnet(args).cuda()
    out = net(data)
    print(out)
