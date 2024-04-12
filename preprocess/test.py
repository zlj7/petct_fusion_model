import os

# 文件夹路径
folder_path = '/data3/share/Shanghai_Pulmonary/sliced_img/pet/'

# 初始化变量
max_file_count = 0
max_file_dir = ''

# 遍历文件夹中的所有子目录
for dir_name in os.listdir(folder_path):
    # 获取子目录的完整路径
    dir_path = os.path.join(folder_path, dir_name)
    
    # 确保这是一个目录而不是文件
    if os.path.isdir(dir_path):
        # 获取目录中的文件数量
        file_count = len(os.listdir(dir_path))
        
        # 如果这个目录的文件数量比之前找到的目录的文件数量多，那么更新最大文件数量和对应的目录
        if file_count > max_file_count:
            max_file_count = file_count
            max_file_dir = dir_path

print(f'文件数量最多的文件夹是：{max_file_dir}，其中包含了{max_file_count}个文件。')
