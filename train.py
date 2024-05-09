import torch.nn as nn
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')

from models.pcfnet import pcfnet,Args
# from models.ct_img import ThreeInOne,Args
# from models.basemodel import ThreeInOne,Args
from dataset.petct_dataset import petct_dataset
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(513)  # 你可以选择任何你喜欢的数字作为种子

weight_decay = 0.0001

epoches = 100

def train(net, train_dataloader, optimizer):
    net.train()
    predict_list, tgt_list = [], []
    loss_t = 0.
    for batch_idx, item in enumerate(train_dataloader):
        ct_img = item['ct_img'].cuda()
        pet_img = item['pet_img'].cuda()
        label = item['label'].cuda()
        pred = net(ct_img, pet_img)
        loss = nn.CrossEntropyLoss()(pred, label)
        loss_t += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict_list += list(pred.data.max(1)[1].cpu().numpy())
        tgt_list += list(label.cpu().numpy())
        

    acc = metrics.accuracy_score(tgt_list, predict_list)
    return acc, loss_t/len(train_dataloader)

def valid(net, val_dataloader):
    net.eval()
    loss_t = 0.
    l2_loss = 0.
    predict_list = []
    tgt_list = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            ct_img = item['ct_img'].cuda()
            pet_img = item['pet_img'].cuda()
            label = item['label'].cuda()
            pred = net(ct_img, pet_img)
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            # L2损失
            for W in net.parameters():
                l2_reg = W.norm(2)
                l2_loss += l2_reg
            predict_list += list(pred.data.max(1)[1].cpu().numpy())
            tgt_list += list(label.cpu().numpy())

    acc = metrics.accuracy_score(tgt_list, predict_list)
    precision = metrics.precision_score(tgt_list, predict_list)
    recall = metrics.recall_score(tgt_list, predict_list)
    f1_score = metrics.f1_score(tgt_list, predict_list)
    auc = metrics.roc_auc_score(tgt_list, predict_list)  # 计算 AUC 指标
    print(f"auc: {auc}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader), l2_loss/len(val_dataloader) * weight_decay
        # loss_t += loss.item()

def save_fig(curve, title):
    plt.figure()
    plt.plot(curve)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    if not os.path.exists("output/original_tiof_petctfuse/"):
        os.makedirs("output/original_tiof_petctfuse/")
    plt.savefig(f'output/original_tiof_petctfuse/{title}.png')  # 保存准确率曲线图像

if __name__ == '__main__':
    train_dataset = petct_dataset('dataset/train.txt', 'train_dataset')
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                                             drop_last=True)

    test_dataset = petct_dataset('dataset/test.txt', 'test_dataset')
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4,
                                             drop_last=True)

    args = Args()
    net = pcfnet(args).cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4,
                                          betas=(0.9, 0.999), weight_decay=weight_decay)
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=10, verbose=True)

    train_acc_curve = []
    test_acc_curve = []
    train_loss_curve = []
    precision_curve = []
    recall_curve = []
    f1_score_curve = []
    test_loss_curve = []
    auc_curve = []
    l2_loss_curve = []

    for epoch in range(epoches):
        train_acc, train_loss = train(net, train_load, optim)
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)

        test_acc, precision, recall, f1_score, auc, test_loss, l2_loss = valid(net, test_load)
        # 更新学习率
        scheduler.step(test_loss)
        test_acc_curve.append(test_acc)
        precision_curve.append(precision)
        recall_curve.append(recall)
        f1_score_curve.append(f1_score)
        test_loss_curve.append(test_loss)
        auc_curve.append(auc)
        l2_loss_curve.append(l2_loss.cpu())

        # 绘制曲线
        save_fig(train_acc_curve, "train_accuracy")
        save_fig(train_loss_curve, "train_loss")
        save_fig(test_loss_curve, "test_loss")
        save_fig(test_acc_curve, "test_accuracy")
        save_fig(precision_curve, "precision")
        save_fig(recall_curve, "recall")
        save_fig(f1_score_curve, "f1_score")
        save_fig(auc_curve, "auc")
        save_fig(l2_loss_curve, "l2_loss")

        print(f"epoch {epoch} finished!")