import numpy as np

# 读取.npy文件
data = np.load('/data3/share/Shanghai_Pulmonary/NCdata_512/train_ct.npy')

# 打印数据的维度
print("数据维度:", data.shape)
