import nibabel as  nb
import skimage
import SimpleITK as sitk
import numpy as np
import os

# def resample(data,ori_space, header, spacing):
#     ### Calculate new space
#     new_width = round(ori_space[0] * header['pixdim'][1] / spacing[0])
#     new_height = round(ori_space[1] * header['pixdim'][2] / spacing[1])
#     new_channel = round(ori_space[2] * header['pixdim'][3] / spacing[2])
#     new_space = [new_width, new_height, new_channel]

#     data_resampled = skimage.transform.resize(data,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)
#     return data_resampled#.transpose(1,2,0)


# img = nb.load('/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/221/221.nii.gz') #读取nii格式文件
# img_affine = img.affine
# data = img.get_data()
# header = img.header

# ### Get original space
# width, height, channel = img.dataobj.shape
# print(width, height, channel)
# ori_space = [width,height,channel]
# # Resample to have 1.0 spacing in all axes
# spacing = [1.0, 1.0, 1.0]
# data_resampled = resample(data,ori_space, header, spacing)
# print(data_resampled.shape)
# # nb.save(data_resampled, './save_image.nii.gz')

# # 创建新的nii图像
# new_image = nb.Nifti1Image(data_resampled, np.eye(4))

# # 保存为.nii.gz文件
# nb.save(new_image, 'nib.nii.gz')

##############################################

path = "/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/"
out_path = "/data3/share/Shanghai_Pulmonary/data/"

# 遍历指定路径下的所有文件
for filename in os.listdir(path):
    # 获取文件名（不含后缀名）
    name, ext = os.path.splitext(filename)
    if(name == "295" or name == "780" or name == "122"):
        continue
    
    print(f"Processing {name}!")
    img = nb.load(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/CT_image_nii/{name}/{name}.nii.gz") #读取nii格式文件

    img_affine = img.affine
    ct_data = img.get_data()
    header = img.header
    ### Get original space
    width, height, channel = img.dataobj.shape
    # print(width, height, channel)
    ori_space = [width,height,channel]
    # Resample to have 1.0 spacing in all axes
    spacing = [1.0, 1.0, 1.0]

    ### Calculate new space
    new_width = round(ori_space[0] * header['pixdim'][1] / spacing[0])
    new_height = round(ori_space[1] * header['pixdim'][2] / spacing[1])
    new_channel = round(ori_space[2] * header['pixdim'][3] / spacing[2])
    new_space = [new_width, new_height, new_channel]
    new_ct_img = skimage.transform.resize(ct_data,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

    # ct_mask = nb.load(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/{name}.nrrd")
    # ct_mask_data = ct_mask.get_data()
    ct_mask = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask CT/{name}.nrrd")
    ct_mask = sitk.Cast(sitk.RescaleIntensity(ct_mask, outputMaximum=2), sitk.sitkUInt8)
    ct_mask = sitk.GetArrayFromImage(ct_mask)
    new_ct_mask = skimage.transform.resize(ct_mask,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

    pet_img = nb.load(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/PET_image_nii/{name}/{name}.nii.gz")
    pet_data = pet_img.get_data()
    new_pet_img = skimage.transform.resize(pet_data,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

    # pet_mask = nb.load(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask PET/{name}.nrrd")
    # pet_mask_data = pet_mask.get_data()
    pet_mask = sitk.ReadImage(f"/data2/share/Shanghai_Pulmonary/PETCT ROI/mask PET/{name}.nrrd")
    pet_mask = sitk.Cast(sitk.RescaleIntensity(pet_mask, outputMaximum=2), sitk.sitkUInt8)
    pet_mask = sitk.GetArrayFromImage(pet_mask)
    new_pet_mask = skimage.transform.resize(pet_mask,new_space,order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)


    # 创建新的nii图像
    new_image = nb.Nifti1Image(new_ct_img, np.eye(4))
    # 保存为.nii.gz文件
    nb.save(new_image, f'/data3/share/Shanghai_Pulmonary/data/ct/{name}.nii.gz')

    # 创建新的nii图像
    new_image = nb.Nifti1Image(new_ct_mask, np.eye(4))
    # 保存为.nii.gz文件
    nb.save(new_image, f'/data3/share/Shanghai_Pulmonary/data/ct_mask/{name}.nii.gz')

    # 创建新的nii图像
    new_image = nb.Nifti1Image(new_pet_img, np.eye(4))
    # 保存为.nii.gz文件
    nb.save(new_image, f'/data3/share/Shanghai_Pulmonary/data/pet/{name}.nii.gz')

    # 创建新的nii图像
    new_image = nb.Nifti1Image(new_pet_mask, np.eye(4))
    # 保存为.nii.gz文件
    nb.save(new_image, f'/data3/share/Shanghai_Pulmonary/data/pet_mask/{name}.nii.gz')

    print(f"Finish processing {name}!")

