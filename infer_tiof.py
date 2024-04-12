import torch.nn as nn
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
import numpy as np
import random
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')


# from models.tiof import ThreeInOne,Args
# from models.ct_img import ThreeInOne,Args
from models.resnet_3channel import Resnet, Resnet_orin, Args
from models.srescnn import srescnn
from dataset.nc_dataset import petct_dataset
from preprocess.channel3_img_dataset import petct_dataset as infer_dataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(712)  # 你可以选择任何你喜欢的数字作为种子

def valid(net, val_dataloader):
    net.eval()
    loss_t = 0.
    l2_loss = 0.
    predict_proba_list = []
    tgt_list = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            ct_img = item['ct_img'].unsqueeze(1).float().cuda()
            pet_img = item['pet_img'].unsqueeze(1).float().cuda()
            # bs = len(channel_3_img)
            label = item['label'].squeeze()
            # label = torch.full((bs,), label_value.item())
            label = label.cuda()
            pred = net(ct_img, pet_img)
            # avg_pred = pred.mean(dim=0, keepdim=True) #求平均
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            # L2损失
            for W in net.parameters():
                l2_reg = W.norm(2)
                l2_loss += l2_reg
            predict_proba_list += list(pred.data[:, 1].cpu().numpy())  # 选择正类的概率
            tgt_list += list(label.cpu().numpy())

    acc = metrics.accuracy_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    precision = metrics.precision_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    recall = metrics.recall_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    f1_score = metrics.f1_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    auc = metrics.roc_auc_score(tgt_list, predict_proba_list)  # 使用概率分数计算 AUC

    print(f"auc: {auc}")
    print(f"f1_score: {f1_score}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader), l2_loss/len(val_dataloader)
        # loss_t += loss.item()

def valid_infer(net, val_dataloader):
    net.eval()
    loss_t = 0.
    predict_proba_list = []
    tgt_list = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            ct_img = item['ct_img'].squeeze(0).unsqueeze(1).float().cuda()
            # print(f'infer:{ct_img.shape}')
            pet_img = item['pet_img'].squeeze(0).unsqueeze(1).float().cuda()
            bs = len(ct_img)
            label_value = item['label']
            label = torch.full((bs,), label_value.item())
            label = label.cuda()
            pred = net(ct_img, pet_img)
            avg_pred = pred.mean(dim=0, keepdim=True) #求平均
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            predict_proba_list += list(avg_pred.data[:, 1].cpu().numpy())  # 选择正类的概率
            tgt_list += list(label_value.cpu().numpy())

    acc = metrics.accuracy_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    precision = metrics.precision_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    recall = metrics.recall_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    f1_score = metrics.f1_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    auc = metrics.roc_auc_score(tgt_list, predict_proba_list)  # 使用概率分数计算 AUC
    
    print(f"patient auc: {auc}")
    print(f"patient f1_score: {f1_score}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader)
        # loss_t += loss.item()

if __name__ == '__main__':
    # 使用文件列表初始化数据集
    # train_ct_files = ['sh_pu_ct_train.npy']
    # train_pet_files = ['sh_pu_pet_train.npy']
    # train_fuse_files = ['sh_pu_fuse_train.npy']
    # train_labels_files = ['sh_pu_label_train.npy']

    # test_ct_files = ['sh_pu_ct_test.npy']
    # test_pet_files = ['sh_pu_pet_test.npy']
    # test_fuse_files = ['sh_pu_fuse_test.npy']
    # test_labels_files = ['sh_pu_label_test.npy']

    train_ct_files = ['sh_pu_ct_train_30.npy']
    train_pet_files = ['sh_pu_pet_train_30.npy']
    train_fuse_files = ['sh_pu_fuse_train_30.npy']
    train_labels_files = ['sh_pu_label_train_30.npy']

    test_ct_files = ['sh_pu_ct_test_30.npy']
    test_pet_files = ['sh_pu_pet_test_30.npy']
    test_fuse_files = ['sh_pu_fuse_test_30.npy']
    test_labels_files = ['sh_pu_label_test_30.npy']
    
    # train_ct_files = ['hebeixptrainct2_64.npy', 'sphxptrainct2_64.npy']
    # train_pet_files = ['hebeixptrainpet2_64.npy', 'sphxptrainpet2_64.npy']
    # train_fuse_files = ['hebeixptrainfuse2_64.npy', 'sphxptrainfuse2_64.npy']
    # train_labels_files = ['hebeiytrain.npy', 'sphytrain.npy']

    # test_ct_files = ['hebeixptestct2_64.npy', 'sphxptestct2_64.npy']
    # test_pet_files = ['hebeixptestpet2_64.npy', 'sphxptestpet2_64.npy']
    # test_fuse_files = ['hebeixptestfuse2_64.npy', 'sphxptestfuse2_64.npy']
    # test_labels_files = ['hebeiytest.npy', 'sphytest.npy']

    test_dataset = petct_dataset(test_ct_files, test_pet_files, test_fuse_files, test_labels_files,train=False)
    print(len(test_dataset))
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4,
                                                drop_last=True)
    
    infer_dataset = infer_dataset('preprocess/test_6.txt', 'test_dataset')
    infer_load = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    model_path = "output/patient_tiof/net.pth"
    # net = Resnet(args).cuda()
    net = torch.load(model_path)
    # net = models.resnet18(pretrained=True)
    # # 如果你只想在最后一层进行训练，你可以设置：
    # for param in net.parameters():
    #     param.requires_grad = False
    # # 然后重新定义最后一层
    # net.fc = nn.Linear(net.fc.in_features, 2)  
    # net = net.cuda()
    # initialize_weights(net)

    test_acc, precision, recall, f1_score, auc, test_loss, l2_loss = valid(net, test_load)
    test_acc, precision, recall, f1_score, auc, test_loss = valid_infer(net, infer_load)