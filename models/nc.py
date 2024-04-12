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

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num + channel_out, out_channels=channel_num, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(channel_num),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num + channel_out, out_channels=channel_num, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(channel_num),
            nn.ReLU())

        self.core = core

    def forward(self, ct_img, pet_img, ct, pet):
        # x.shape = yi.shape = bs*N*H*W
        if self.core == 'img':
            # print(f"ct: {ct.shape}")
            # print(f"pet: {pet.shape}")
            ct, ct_squeeze = self.se1(ct_img, ct)
            pet, pet_squeeze = self.se2(pet_img, pet)
            ct_img = torch.cat([ct_img, ct_squeeze], dim=1)
            ct_img = self.conv1(ct_img)
            pet_img = torch.cat([pet_img, pet_squeeze], dim=1)
            pet_img = self.conv2(pet_img)

        return ct_img, pet_img, ct, pet


class tsb_layer(nn.Module):
    def __init__(self, ct_img, pet_img, ct, pet, current_layer, fuse_layer, channel_num, spatial_dim, channel_out=4, core='img'):
        super(tsb_layer, self).__init__()
        self.featrue_ct_img = ct_img
        self.featrue_pet_img = pet_img
        self.featrue_ct = ct
        self.featrue_pet = pet

        self.apply_tsb = current_layer >= fuse_layer
        if self.apply_tsb:
            self.layer = tsb(channel_num, spatial_dim, channel_out, core)

    def forward(self, input):
        ct_img, pet_img, ct, pet = input
        ct_img = self.featrue_ct_img(ct_img)
        pet_img = self.featrue_pet_img(pet_img)
        ct = self.featrue_ct(ct)
        pet = self.featrue_pet(pet)
        if self.apply_tsb:
            ct_img, pet_img, ct, pet = self.layer(ct_img, pet_img, ct, pet)
        return ct_img, pet_img, ct, pet


ResNet_list = {'resnet50': resnet50, 'resnet34': resnet34, 'resnet18': resnet18}
res_channels = {'resnet50': [64, 256, 512, 1024, 2048],
                'resnet34': [64, 64, 128, 256, 512],
                'resnet18': [64, 64, 128, 256, 512]}


class ThreeInOne(nn.Module):
    def __init__(self, args):
        super(ThreeInOne, self).__init__()
        num_classes = args.num_classes
        channel_out = 4
        fuse_layer = 3
        core = 'img'
        print('Core Component is {}'.format(core))


        self.backbone_ct_img = resnet50(pretrained=True)
        self.backbone_ct_img.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))
        self.backbone_pet_img = resnet50(pretrained=True)
        self.backbone_pet_img.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))
        self.backbone_ct = resnet50(pretrained=True)
        self.backbone_ct.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))
        self.backbone_pet = resnet50(pretrained=True)
        self.backbone_pet.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))


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
        self.spatial_dim = [256 ** 2, 128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2]
        self.tsb = nn.Sequential(*[
            tsb_layer(self.feature_ct_img[i], self.feature_pet_img[i], self.feature_ct[i], self.feature_pet[i], i,
                      fuse_layer, self.channel_num[i], self.spatial_dim[i], channel_out, core)
            for i in range(len(self.channel_num))])


        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc_ct = nn.Linear(res_channels['resnet50'][-1]*2, res_channels['resnet50'][-1]*2, bias=True)
        self.fc_pet = nn.Linear(res_channels['resnet50'][-1]*2, res_channels['resnet50'][-1]*2, bias=True)
        self.fc_final = nn.Linear(res_channels['resnet50'][-1]*2, num_classes, bias=True)
        # 添加 dropout 层
        self.dropout = nn.Dropout(p=0.5)

        self.name = 'tiof'
        self.lossfn = nn.CrossEntropyLoss()

    def model_name(self):
        return self.name


    def forward(self, ct_img, pet_img):
        bs = ct_img.size(0)
        # ct_img, ct = torch.split(ct_data, [1, 1], dim=1)
        # pet_img, pet = torch.split(pet_data, [1, 1], dim=1)

        ct_img, pet_img, ct, pet = self.tsb([ct_img, pet_img, ct_img, pet_img,])

        ct_img, pet_img, ct, pet = self.pool(ct_img), self.pool(pet_img), self.pool(ct), self.pool(pet)
        x_ct = torch.cat([ct_img.squeeze(dim=2).squeeze(dim=2),
                       ct.squeeze(dim=2).squeeze(dim=2)], dim=-1)
        x_ct = self.fc_ct(x_ct)
        x_ct = self.dropout(x_ct)  # 在全连接层之后添加 dropout
        # x_pet = torch.cat([pet_img.squeeze(dim=2).squeeze(dim=2),
        #                pet.squeeze(dim=2).squeeze(dim=2)], dim=-1)
        # x_pet = self.fc_pet(x_pet)
        # x = torch.cat([x_ct, x_pet], dim=1)
        # # print(f"x_ct: {x_ct.shape}")
        # # print(f"x_pet: {x_pet.shape}")
        # # print(f"x: {x.shape}")
        x = self.fc_final(x_ct)
        # x = torch.sigmoid(x)

        # loss = self.lossfn(x, tgt)

        return x #, loss

class Args():
    def __init__(self):
        self.num_classes = 2


if __name__ == '__main__':
    args = Args()
    ct_data = torch.rand(1, 1, 64, 64).cuda()
    pet_data = torch.rand(1, 1, 64, 64).cuda()
    tgt = torch.tensor([1], dtype=torch.long).cuda()
    net = ThreeInOne(args).cuda()
    out = net(ct_data, pet_data)
    print(out)
