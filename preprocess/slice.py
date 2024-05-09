import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import cv2

def process_images(ct, ct_mask, pet, pet_mask, filename, save_path):
    # 对每个mask进行操作
    for i in range(ct_mask.shape[2]):
        # 获取mask
        ct_mask_slice = ct_mask[:,:,i]
        pet_mask_slice = pet_mask[:,:,i]

        # 找到mask的最小矩形
        # print(ct_mask_slice.shape)
        ct_x, ct_y, ct_w, ct_h = cv2.boundingRect(ct_mask_slice)
        pet_x, pet_y, pet_w, pet_h = cv2.boundingRect(pet_mask_slice)

        # if(ct_w * ct_h < 0.3 * ((ct_w + 20) * (ct_h +20)) or pet_w * pet_h < 0.3 * ((pet_w + 20) * (pet_h +20))):
        if(ct_w * ct_h < 1 or pet_w * pet_h < 1):
            continue

        # 扩展矩形的宽高
        ct_x = max(0, ct_x - 10)
        ct_y = max(0, ct_y - 10)
        ct_w = min(ct.shape[1], ct_w + 20)
        ct_h = min(ct.shape[0], ct_h + 20)

        pet_x = max(0, pet_x - 10)
        pet_y = max(0, pet_y - 10)
        pet_w = min(pet.shape[1], pet_w + 20)
        pet_h = min(pet.shape[0], pet_h + 20)

        # 切下对应位置的图像
        ct_slice = ct[ct_y:ct_y+ct_h, ct_x:ct_x+ct_w, i]
        pet_slice = pet[pet_y:pet_y+pet_h, pet_x:pet_x+pet_w, i]
        # print(ct_slice)
        # print(pet_slice)

        os.makedirs(f"{save_path}/ct/{filename}/", exist_ok=True)
        os.makedirs(f"{save_path}/pet/{filename}/", exist_ok=True)
        # 保存灰度图
        cv2.imwrite(f'{save_path}/ct/{filename}/{i}.png', ct_slice)
        cv2.imwrite(f'{save_path}/pet/{filename}/{i}.png', pet_slice)

# # CT数据及mask
# ct_img = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/data/ct/619.nii.gz')
# ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkUInt8)
# ct_img = sitk.GetArrayFromImage(ct_img).transpose(1,2,0)
# print(f"ct_img:{ct_img.shape}")

# ct_mask = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/data/ct_mask/619.nii.gz')
# ct_mask = sitk.Cast(sitk.RescaleIntensity(ct_mask, outputMaximum=2), sitk.sitkUInt8)
# ct_mask = sitk.GetArrayFromImage(ct_mask).transpose(1,2,0)
# print(f"ct_mask:{ct_mask.shape}")

# # PET数据及mask
# pet_img = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/data/pet/619.nii.gz')
# pet_img = sitk.Cast(sitk.RescaleIntensity(pet_img), sitk.sitkUInt8)
# pet_img = sitk.GetArrayFromImage(pet_img).transpose(1,2,0)
# print(f"pet_img:{pet_img.shape}")

# pet_mask = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/data/pet_mask/619.nii.gz')
# pet_mask = sitk.Cast(sitk.RescaleIntensity(pet_mask, outputMaximum=2), sitk.sitkUInt8)
# pet_mask = sitk.GetArrayFromImage(pet_mask).transpose(1,2,0)
# print(f"pet_mask:{pet_mask.shape}")

# process_images(ct_img, ct_mask, pet_img, pet_mask, "./out")

path = "/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏_resample/ct/"
# 遍历指定路径下的所有文件
for filename in os.listdir(path):
    # 获取文件名（不含后缀名）
    name = filename.split('.')#os.path.splitext(filename)
    # print(name[0])
    if(name[0] == "295" or name[0] == "730" or name[0] == "738"):
        continue
    # CT数据及mask
    ct_img = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏_resample/ct/{name[0]}.nii.gz')
    ct_img = sitk.Cast(sitk.RescaleIntensity(ct_img), sitk.sitkUInt8)
    ct_img = sitk.GetArrayFromImage(ct_img).transpose(1,2,0)
    # print(f"ct_img:{ct_img.shape}")

    ct_mask = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏_resample/ct_mask/{name[0]}.nii.gz')
    ct_mask = sitk.Cast(sitk.RescaleIntensity(ct_mask, outputMaximum=2), sitk.sitkUInt8)
    ct_mask = sitk.GetArrayFromImage(ct_mask).transpose(1,2,0)
    # print(f"ct_mask:{ct_mask.shape}")

    # PET数据及mask
    pet_img = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏_resample/pet/{name[0]}.nii.gz')
    pet_img = sitk.Cast(sitk.RescaleIntensity(pet_img), sitk.sitkUInt8)
    pet_img = sitk.GetArrayFromImage(pet_img).transpose(1,2,0)
    # print(f"pet_img:{pet_img.shape}")

    pet_mask = sitk.ReadImage(f'/data3/share/Shanghai_Pulmonary/PET 2nd 脱敏_resample/pet_mask/{name[0]}.nii.gz')
    pet_mask = sitk.Cast(sitk.RescaleIntensity(pet_mask, outputMaximum=2), sitk.sitkUInt8)
    pet_mask = sitk.GetArrayFromImage(pet_mask).transpose(1,2,0)
    # print(f"pet_mask:{pet_mask.shape}")

    # print(type(ct_mask))
    process_images(ct_img, ct_mask, pet_img, pet_mask, name[0], "/data3/share/Shanghai_Pulmonary/sliced_img_no_filter/")
    print(f"Finish processing {name[0]}!")