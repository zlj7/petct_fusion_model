import torch.nn as nn
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import os

from models.resnet_3channel import Resnet,Args
from preprocess.channel3_img_dataset import petct_dataset

def valid(net, val_dataloader):
    net.eval()
    loss_t = 0.
    predict_list = []
    tgt_list = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            channel_3_img = item['channel_3_img'].squeeze(0).float().cuda()
            bs = len(channel_3_img)
            label_value = item['label']
            label = torch.full((bs,), label_value.item())
            label = label.cuda()
            pred = net(channel_3_img)
            avg_pred = pred.mean(dim=0, keepdim=True) #求平均
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            predict_list += list(avg_pred.data.max(1)[1].cpu().numpy())
            tgt_list += list(label_value.cpu().numpy())

    acc = metrics.accuracy_score(tgt_list, predict_list)
    precision = metrics.precision_score(tgt_list, predict_list)
    recall = metrics.recall_score(tgt_list, predict_list)
    f1_score = metrics.f1_score(tgt_list, predict_list)
    auc = metrics.roc_auc_score(tgt_list, predict_list)  # 计算 AUC 指标
    print(f"acc: {acc}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader)


# 加载模型权重
args = Args()
net = Resnet(args).cuda()
net.load_state_dict(torch.load('net_all.pth'))

test_dataset = petct_dataset('preprocess/train_6.txt', 'test_dataset')
test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

test_acc, precision, recall, f1_score, auc, test_loss = valid(net, test_load)
print(test_acc)