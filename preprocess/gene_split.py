import os
import random

ct_folder = "/data3/share/Shanghai_Pulmonary/sliced_img_30check/ct/"
pet_folder = "/data3/share/Shanghai_Pulmonary/sliced_img_30check/pet/"
output_train = "train.txt"
output_test = "test.txt"

# 获取所有ct文件夹的路径
ct_paths = [os.path.join(ct_folder, folder_name) for folder_name in os.listdir(ct_folder)]
random.shuffle(ct_paths)  # 打乱顺序

# 划分训练集和测试集的索引
train_size = int(0.8 * len(ct_paths))
train_paths = ct_paths[:train_size]
test_paths = ct_paths[train_size:]

# 写入训练集文件路径
with open(output_train, "w") as file:
    for ct_path in train_paths:
        folder_name = os.path.basename(ct_path)
        pet_path = os.path.join(pet_folder, folder_name)
        
        if os.path.exists(pet_path):
            for file_name in os.listdir(ct_path):
                if file_name.endswith(".png"):
                    ct_file_path = os.path.join(ct_path, file_name)
                    pet_file_path = os.path.join(pet_path, file_name)
                    
                    if os.path.exists(pet_file_path):
                        file.write(pet_file_path + "\n")

# 写入测试集文件路径
with open(output_test, "w") as file:
    for ct_path in test_paths:
        folder_name = os.path.basename(ct_path)
        pet_path = os.path.join(pet_folder, folder_name)
        
        if os.path.exists(pet_path):
            for file_name in os.listdir(ct_path):
                if file_name.endswith(".png"):
                    ct_file_path = os.path.join(ct_path, file_name)
                    pet_file_path = os.path.join(pet_path, file_name)
                    
                    if os.path.exists(pet_file_path):
                        file.write(pet_file_path + "\n")