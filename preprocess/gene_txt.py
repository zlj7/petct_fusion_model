import os
import random

# 指定路径
path = '/data3/share/Shanghai_Pulmonary/sliced_img_30/ct/'

# 获取路径下的所有子文件夹
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

# 打乱文件夹的顺序
random.shuffle(folders)

# 计算训练集和测试集的大小
train_size = int(len(folders) * 0.8)
test_size = len(folders) - train_size

# 分割训练集和测试集
train_folders = folders[:train_size]
# train_folders = folders
test_folders = folders[train_size:]

# 将训练集写入train.txt
with open('train.txt', 'w') as f:
    for folder in train_folders:
        f.write(folder + '\n')

# 将测试集写入test.txt
with open('test.txt', 'w') as f:
    for folder in test_folders:
        f.write(folder + '\n')


# import pandas as pd

# # 读取Excel文件
# df = pd.read_excel("/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET临床信息-睿医-2024-4-24.xlsx")

# # 找到“数据来源”列为“LungMate-001”“LungMate-002”“LungMate-003”“LungMate-009”“LungMate-017”的行的“影像组学序列号”列
# train_lungmate = df[df["数据来源"].isin(["LungMate-001", "LungMate-002", "LungMate-003", "LungMate-009", "LungMate-017"])]["影像组学序列号"]

# # 按行写入“train_lungmate.txt”
# with open("train_lungmate.txt", "w") as file:
#     for item in train_lungmate:
#         file.write(str(item) + "\n")

# # 找到“数据来源”列为“LungMate-013”“手术回顾库”“非手术回顾库”的“影像组学序列号”列
# test_lungmate = df[df["数据来源"].isin(["LungMate-013", "手术回顾库", "非手术回顾库"])]["影像组学序列号"]

# # 按行写入“text_lungmate.txt”
# with open("test_lungmate.txt", "w") as file:
#     for item in test_lungmate:
#         file.write(str(item) + "\n")
