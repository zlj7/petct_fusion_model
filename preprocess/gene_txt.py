import os
import random

# 指定路径
path = '/data3/share/Shanghai_Pulmonary/sliced_img_30/ct/'

# 获取路径下的所有子文件夹
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

# 打乱文件夹的顺序
random.shuffle(folders)

# 计算训练集和测试集的大小
train_size = int(len(folders))
test_size = len(folders) - train_size

# 分割训练集和测试集
# train_folders = folders[:train_size]
train_folders = folders
test_folders = folders[train_size:]

# 将训练集写入train.txt
with open('train_all.txt', 'w') as f:
    for folder in train_folders:
        f.write(folder + '\n')

# 将测试集写入test.txt
with open('test_all.txt', 'w') as f:
    for folder in test_folders:
        f.write(folder + '\n')
