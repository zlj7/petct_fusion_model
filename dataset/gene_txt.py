import pandas as pd
import os
import random

# 生成train.txt和test.txt，每行一个样本
path = "/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/"

names = []
# 遍历指定路径下的所有文件
for filename in os.listdir(path):
    # 获取文件名（不含后缀名）
    name, ext = os.path.splitext(filename)
    if(name == "295"):
        continue
    
    df = pd.read_excel('/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET临床信息-睿医.xlsx') # 读取.xlsx文件
    recist = df.loc[df['影像组学序列号'] == int(name), 'RECIST'].values[0] # 查询RECIST
    if(recist == 'CR' or recist == 'PR' or recist == 'SD' or recist == 'PD'):
        # 将文件名添加到列表中
        names.append(name)

path = "/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/CT_MASK/"

# 遍历指定路径下的所有文件
for filename in os.listdir(path):
    # 获取文件名（不含后缀名）
    name, ext = os.path.splitext(filename)
    if(name == "295" or name == "780" or name == "122"):
        continue
    
    df = pd.read_excel('/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏/PET临床信息-睿医.xlsx') # 读取.xlsx文件
    recist = df.loc[df['影像组学序列号'] == int(name), 'RECIST'].values[0] # 查询RECIST
    if(recist == 'CR' or recist == 'PR' or recist == 'SD' or recist == 'PD'):
        # 将文件名添加到列表中
        names.append(name)

# 随机打乱样本名列表
random.shuffle(names)

# 计算分割点
split_point = int(len(names) * 0.8)

# 分割成训练集和测试集
train_set = names[:split_point]
test_set = names[split_point:]

# 将样本名写入train.txt
with open("train.txt", "w") as train_file:
    for sample_name in train_set:
        train_file.write(sample_name + "\n")

# 将样本名写入test.txt
with open("test.txt", "w") as test_file:
    for sample_name in test_set:
        test_file.write(sample_name + "\n")