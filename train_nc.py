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


# from models.ct_img import ThreeInOne,Args
# from models.resnet_3channel import Resnet, Resnet_orin
from models.srescnn import srescnn
from models.nc_3channel import pcfnet, Args
from dataset.nc_dataset import petct_dataset
from dataset.patient_dataset import petct_dataset as infer_dataset
import wandb
# from tensorboardX import SummaryWriter

# writer = SummaryWriter()

use_wandb = False

if use_wandb:
    wandb.init(project='sh_pu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(712)  

epoches = 100

weight_decay = 0.001

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def train(net, train_dataloader, optimizer):
    net.train()
    predict_list, tgt_list = [], []
    predict_proba_list = []
    loss_t = 0.
    for batch_idx, item in enumerate(train_dataloader):
        channel_3_img = item['channel_3_img'].float().cuda()
        label = item['label'].squeeze()
        label = label.cuda()
        pred = net(channel_3_img)
        loss = nn.CrossEntropyLoss()(pred, label)
        loss_t += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict_proba_list += [pred.data.cpu().numpy()]  # 选择所有类别的概率
        tgt_list += list(label.cpu().numpy())
    
    predict_proba_list = np.concatenate(predict_proba_list, axis=0)
    # print(predict_proba_list.shape)
    # 将概率列表转换为类别预测
    predict_list = np.argmax(predict_proba_list, axis=1)

    # 计算各项指标
    acc = metrics.accuracy_score(tgt_list, predict_list)
    return acc, loss_t/len(train_dataloader)


best_loss = 1e9
best_auc = 0.
best_f1 = 0.
no_improve = 0

def valid(net, val_dataloader, save_path):
    net.eval()
    loss_t = 0.
    l2_loss = 0.
    predict_proba_list = []
    tgt_list = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            channel_3_img = item['channel_3_img'].float().cuda()
            label = item['label'].squeeze()
            label = label.cuda()
            pred = net(channel_3_img)
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            # L2损失
            for W in net.parameters():
                l2_reg = W.norm(2)
                l2_loss += l2_reg
            softmax = nn.Softmax(dim=1)
            probas = softmax(pred).data.cpu().numpy()
            predict_proba_list += [probas]  # 选择所有类别的概率
            tgt_list += list(label.cpu().numpy())

    predict_proba_list = np.concatenate(predict_proba_list, axis=0)
    # print(predict_proba_list.shape)
    # 将概率列表转换为类别预测
    predict_list = np.argmax(predict_proba_list, axis=1)

    # 计算各项指标
    acc = metrics.accuracy_score(tgt_list, predict_list)
    precision = metrics.precision_score(tgt_list, predict_list, average='macro')
    recall = metrics.recall_score(tgt_list, predict_list, average='macro')
    f1_score = metrics.f1_score(tgt_list, predict_list, average='macro')

    # 对于多分类AUC，需要使用一对多或一对一的方法
    # 假设你有4个类别
    # labels = list(range(4))
    # auc = metrics.roc_auc_score(tgt_list, predict_proba_list, multi_class='ovo', labels=labels)
    num_classes = len(np.unique(tgt_list))
    # if num_classes == 2:
    auc = metrics.roc_auc_score(tgt_list, predict_proba_list[:, 1])
    # # else:
    # labels = list(range(4))
    # # print(tgt_list)
    # # print(predict_proba_list)
    # auc = metrics.roc_auc_score(tgt_list, predict_proba_list, multi_class='ovo', labels=labels)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'{save_path}auc.txt', 'a') as f:
        print(auc, file=f)
    print(f"auc: {auc}")
    global best_loss
    global best_auc
    global best_f1
    global no_improve
    if loss < best_loss:
        best_loss = loss
        no_improve = 0
        best_auc = auc
        best_f1 = f1_score
        torch.save(net, f'{save_path}net.pth')
    else:
        no_improve += 1

    # 如果30个epoch都没有提升，就停止训练
    if no_improve >= 30:
        print("Early stopping")
        with open(f'{save_path}best_result.txt', 'w') as f:

            print(f'best_auc: {best_auc}', file=f)
            print(f'best_f1: {best_f1}', file=f)
        exit()

    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader), l2_loss/len(val_dataloader)

def valid_infer(net, val_dataloader, save_path):
    net.eval()
    loss_t = 0.
    predict_proba_list = []
    tgt_list = []
    names = []
    with torch.no_grad():
        for batch_idx, item in enumerate(val_dataloader):
            channel_3_img = item['channel_3_img'].squeeze(0).float().cuda()
            name = item['patient']
            names.append(name)
            bs = len(channel_3_img)
            label_value = item['label']
            label = torch.full((bs,), label_value.item())
            label = label.cuda()
            pred = net(channel_3_img)
            # print(pred.shape)
            avg_pred = pred.mean(dim=0, keepdim=True) #求平均
            loss = nn.CrossEntropyLoss()(pred, label)
            loss_t += loss.item()
            predict_proba_list += list(avg_pred.data[:, 1].cpu().numpy())  # 选择正类的概率
            tgt_list += list(label_value.cpu().numpy())

    acc = metrics.accuracy_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list])
    precision = metrics.precision_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list], average='macro')
    recall = metrics.recall_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list], average='macro')
    f1_score = metrics.f1_score(tgt_list, [1 if x > 0.5 else 0 for x in predict_proba_list], average='macro')
    auc = metrics.roc_auc_score(tgt_list, predict_proba_list, multi_class='ovo')  # 使用概率分数计算 AUC
    with open(f'{save_path}predict_proba_list.txt', 'w') as f:
        print(predict_proba_list, file=f)
    with open(f'{save_path}names.txt', 'w') as f:
        print(names, file=f)
    with open(f'{save_path}pred.txt', 'w') as f:
        print(pred, file=f)
    with open(f'{save_path}patient_auc.txt', 'a') as f:
        print(auc, file=f)
    print(f"patient auc: {auc}")
    # if acc > best_acc:
    #     best_acc = acc
    return acc, precision, recall, f1_score, auc, loss_t/len(val_dataloader)
        # loss_t += loss.item()

def save_fig(save_path, curve, title):
    plt.figure()
    plt.plot(curve)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}{title}.png')  # 保存准确率曲线图像

if __name__ == '__main__':
    fold = 5
    # 使用文件列表初始化数据集
    # train_ct_files = ['sh_pu_ct_train_30.npy']
    # train_pet_files = ['sh_pu_pet_train_30.npy']
    # train_fuse_files = ['sh_pu_fuse_train_30.npy']
    # train_labels_files = ['sh_pu_label_train_30.npy']

    # test_ct_files = ['sh_pu_ct_test_30.npy']
    # test_pet_files = ['sh_pu_pet_test_30.npy']
    # test_fuse_files = ['sh_pu_fuse_test_30.npy']
    # test_labels_files = ['sh_pu_label_test_30.npy']

    train_ct_files = [f'sh_pu_ct_train_all_fold_{fold}.npy']
    train_pet_files = [f'sh_pu_pet_train_all_fold_{fold}.npy']
    train_fuse_files = [f'sh_pu_fuse_train_all_fold_{fold}.npy']
    train_labels_files = [f'sh_pu_label_train_all_fold_{fold}.npy']

    test_ct_files = [f'sh_pu_ct_test_all_fold_{fold}.npy']
    test_pet_files = [f'sh_pu_pet_test_all_fold_{fold}.npy']
    test_fuse_files = [f'sh_pu_fuse_test_all_fold_{fold}.npy']
    test_labels_files = [f'sh_pu_label_test_all_fold_{fold}.npy']

    # train_ct_files = [f'sh_pu_ct_train_50_2cls_lungmate.npy']
    # train_pet_files = [f'sh_pu_pet_train_50_2cls_lungmate.npy']
    # train_fuse_files = [f'sh_pu_fuse_train_50_2cls_lungmate.npy']
    # train_labels_files = [f'sh_pu_label_train_50_2cls_lungmate.npy']

    # test_ct_files = [f'sh_pu_ct_test_50_2cls_lungmate.npy']
    # test_pet_files = [f'sh_pu_pet_test_50_2cls_lungmate.npy']
    # test_fuse_files = [f'sh_pu_fuse_test_50_2cls_lungmate.npy']
    # test_labels_files = [f'sh_pu_label_test_50_2cls_lungmate.npy']

    # train_ct_files = [f'sh_pu_ct_train_30_2cls_lungmate.npy']
    # train_pet_files = [f'sh_pu_pet_train_30_2cls_lungmate.npy']
    # train_fuse_files = [f'sh_pu_fuse_train_30_2cls_lungmate.npy']
    # train_labels_files = [f'sh_pu_label_train_30_2cls_lungmate.npy']

    # test_ct_files = [f'sh_pu_ct_test_30_2cls_lungmate.npy']
    # test_pet_files = [f'sh_pu_pet_test_30_2cls_lungmate.npy']
    # test_fuse_files = [f'sh_pu_fuse_test_30_2cls_lungmate.npy']
    # test_labels_files = [f'sh_pu_label_test_30_2cls_lungmate.npy']

    # train_ct_files = [f'sh_pu_ct_train_30_4cls_fold_{fold}.npy']
    # train_pet_files = [f'sh_pu_pet_train_30_4cls_fold_{fold}.npy']
    # train_fuse_files = [f'sh_pu_fuse_train_30_4cls_fold_{fold}.npy']
    # train_labels_files = [f'sh_pu_label_train_30_4cls_fold_{fold}.npy']

    # test_ct_files = [f'sh_pu_ct_test_30_4cls_fold_{fold}.npy']
    # test_pet_files = [f'sh_pu_pet_test_30_4cls_fold_{fold}.npy']
    # test_fuse_files = [f'sh_pu_fuse_test_30_4cls_fold_{fold}.npy']
    # test_labels_files = [f'sh_pu_label_test_30_4cls_fold_{fold}.npy']
    
    # train_ct_files = ['hebeixptrainct2_64.npy', 'sphxptrainct2_64.npy']
    # train_pet_files = ['hebeixptrainpet2_64.npy', 'sphxptrainpet2_64.npy']
    # train_fuse_files = ['hebeixptrainfuse2_64.npy', 'sphxptrainfuse2_64.npy']
    # train_labels_files = ['hebeiytrain.npy', 'sphytrain.npy']

    # test_ct_files = ['hebeixptestct2_64.npy', 'sphxptestct2_64.npy']
    # test_pet_files = ['hebeixptestpet2_64.npy', 'sphxptestpet2_64.npy']
    # test_fuse_files = ['hebeixptestfuse2_64.npy', 'sphxptestfuse2_64.npy']
    # test_labels_files = ['hebeiytest.npy', 'sphytest.npy']

    dataset = petct_dataset(train_ct_files, train_pet_files, train_fuse_files, train_labels_files)
    print(len(dataset))
    train_load = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4,
                                                drop_last=True)

    for batch_idx, item in enumerate(train_load):
        ct_mean = item['ct_mean']
        ct_std = item['ct_std']
        pet_mean = item['pet_mean']
        pet_std = item['pet_std']
        fuse_mean = item['fuse_mean']
        fuse_std = item['fuse_std']
        channel_mean = item['channel_mean']
        channel_std = item['channel_std']
        # print(f'channel_mean: {channel_mean}')
        break
    
    test_dataset = petct_dataset(test_ct_files, test_pet_files, test_fuse_files, test_labels_files, ct_mean[0], ct_std[0], pet_mean[0], pet_std[0], fuse_mean[0], fuse_std[0], channel_mean[0], channel_std[0],train=False)
    print(len(test_dataset))
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4,
                                                drop_last=True)
    
    # patient_dataset = infer_dataset(f"/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_ct_test_all_patient_fold_{fold}.pkl",ct_mean[0], ct_std[0], pet_mean[0], pet_std[0], fuse_mean[0], fuse_std[0], channel_mean[0], channel_std[0])
    patient_dataset = infer_dataset(f"/data3/share/Shanghai_Pulmonary/NCdata/sh_pu_ct_test_patient.pkl",ct_mean[0], ct_std[0], pet_mean[0], pet_std[0], fuse_mean[0], fuse_std[0], channel_mean[0], channel_std[0])
    
    infer_load = torch.utils.data.DataLoader(patient_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             drop_last=True)

    save_path = f"output_fold/tiof_all_2cls/fold_{fold}/"
    args = Args()
    # net = srescnn(num_classes=2).cuda()
    net = pcfnet(args).cuda()
    # net = models.resnet50(pretrained=False)
    # # 然后重新定义最后一层
    # net.fc = nn.Linear(net.fc.in_features, 2)  
    # net = net.cuda()
    # initialize_weights(net)
    # if use_wandb:
    #     wandb.watch(net, log='all')
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
    auc_curve = []
    test_loss_curve = []
    l2_loss_curve = []

    infer_test_acc_curve = []
    infer_precision_curve = []
    infer_recall_curve = []
    infer_f1_score_curve = []
    infer_auc_curve = []
    infer_test_loss_curve = []

    if os.path.isfile(f'{save_path}auc.txt'):
        os.remove(f'{save_path}auc.txt')

    if os.path.isfile(f'{save_path}patient_auc.txt'):
        os.remove(f'{save_path}patient_auc.txt')


    for epoch in range(epoches):
        train_acc, train_loss = train(net, train_load, optim)
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)
        # for name, param in net.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)

        test_acc, precision, recall, f1_score, auc, test_loss, l2_loss = valid(net, test_load, save_path)
        if use_wandb:
            wandb.log({"train_acc": train_acc, "train_loss": train_loss, "test_acc": test_acc, "precision": precision, 
            "recall": recall, "f1_score": f1_score, "auc": auc, "test_loss": test_loss, "l2_loss": l2_loss * weight_decay})
        # 更新学习率
        scheduler.step(test_loss)
        test_acc_curve.append(test_acc)
        precision_curve.append(precision)
        recall_curve.append(recall)
        f1_score_curve.append(f1_score)
        auc_curve.append(auc)
        test_loss_curve.append(test_loss)
        l2_loss_curve.append(l2_loss.cpu().numpy() * weight_decay)

        # 绘制曲线
        save_fig(save_path, train_acc_curve, "train_accuracy")
        save_fig(save_path, train_loss_curve, "train_loss")
        save_fig(save_path, test_loss_curve, "test_loss")
        save_fig(save_path, test_acc_curve, "test_accuracy")
        save_fig(save_path, precision_curve, "precision")
        save_fig(save_path, recall_curve, "recall")
        save_fig(save_path, f1_score_curve, "f1_score")
        save_fig(save_path, auc_curve, "auc")
        save_fig(save_path, l2_loss_curve, "l2_loss")

        test_acc, precision, recall, f1_score, auc, test_loss = valid_infer(net, infer_load, save_path)
        infer_test_acc_curve.append(test_acc)
        infer_precision_curve.append(precision)
        infer_recall_curve.append(recall)
        infer_f1_score_curve.append(f1_score)
        infer_auc_curve.append(auc)
        infer_test_loss_curve.append(test_loss)

        # 绘制曲线
        save_fig(save_path, infer_test_loss_curve, "patient_test_loss")
        save_fig(save_path, infer_test_acc_curve, "patient_test_accuracy")
        save_fig(save_path, infer_precision_curve, "patient_precision")
        save_fig(save_path, infer_recall_curve, "patient_recall")
        save_fig(save_path, infer_f1_score_curve, "patient_f1_score")
        save_fig(save_path, infer_auc_curve, "patient_auc")



        print(f"epoch {epoch} finished!")
    with open(f'{save_path}best_result.txt', 'w') as f:
        print(f'best_auc: {best_auc}', file=f)
        print(f'best_f1: {best_f1}', file=f)