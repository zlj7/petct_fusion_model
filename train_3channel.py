import torch.nn as nn
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# from models.tiof import ThreeInOne,Args
# from models.ct_img import ThreeInOne,Args
from models.resnet_3channel import Resnet,Args
from preprocess.channel3_img_dataset import petct_dataset

epoches = 100

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(712)  # 你可以选择任何你喜欢的数字作为种子

epoches = 100

weight_decay = 0.001

def train(net, train_dataloader, optimizer):
    net.train()
    predict_list, tgt_list = [], []
    loss_t = 0.
    for batch_idx, item in enumerate(train_dataloader):
        channel_3_img = item['channel_3_img'].squeeze(0).float().cuda()
        bs = len(channel_3_img)
        label_value = item['label']
        label = torch.full((bs,), label_value.item())
        label = label.cuda()
        pred = net(channel_3_img)
        # print(f'pred:{pred.shape}')
        # print(f'label:{label.shape}')
        avg_pred = pred.mean(dim=0, keepdim=True) #求平均
        loss = nn.CrossEntropyLoss()(pred, label)
        loss_t += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict_list += list(avg_pred.data.max(1)[1].cpu().numpy())
        tgt_list += list(label_value.cpu().numpy())
        

    torch.save(net.state_dict(), 'net_all.pth')
    acc = metrics.accuracy_score(tgt_list, predict_list)
    return acc, loss_t/len(train_dataloader)

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
    print(f"auc: {auc}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader)
        # loss_t += loss.item()

def save_fig(curve, title):
    plt.figure()
    plt.plot(curve)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    if not os.path.exists("output/alpha_fusion_3channel_fold_7/"):
        os.makedirs("output/alpha_fusion_3channel_fold_7/")
    plt.savefig(f'output/alpha_fusion_3channel_fold_7/{title}.png')  # 保存准确率曲线图像

if __name__ == '__main__':
    train_dataset = petct_dataset('preprocess/train_6.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    test_dataset = petct_dataset('preprocess/test_6.txt', 'test_dataset')
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    args = Args()
    net = Resnet(args).cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5,
                                          betas=(0.9, 0.999), weight_decay=weight_decay)
    train_acc_curve = []
    test_acc_curve = []
    train_loss_curve = []
    precision_curve = []
    recall_curve = []
    f1_score_curve = []
    auc_curve = []
    test_loss_curve = []

    for epoch in range(epoches):
        train_acc, train_loss = train(net, train_load, optim)
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)

        test_acc, precision, recall, f1_score, auc, test_loss = valid(net, test_load)
        test_acc_curve.append(test_acc)
        precision_curve.append(precision)
        recall_curve.append(recall)
        f1_score_curve.append(f1_score)
        test_loss_curve.append(test_loss)
        auc_curve.append(auc)

        # 绘制曲线
        save_fig(train_acc_curve, "train_accuracy")
        save_fig(train_loss_curve, "train_loss")
        save_fig(test_loss_curve, "test_loss")
        save_fig(test_acc_curve, "test_accuracy")
        save_fig(precision_curve, "precision")
        save_fig(recall_curve, "recall")
        save_fig(f1_score_curve, "f1_score")
        save_fig(auc_curve, "auc")

        print(f"epoch {epoch} finished!")