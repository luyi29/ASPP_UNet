import cv2
import numpy as np
import os #遍历文件夹
import nibabel as nib
import imageio #转换成图像


#生成图片分层的代码
def nii_to_image(filepath,newfilepath,flag="image"):
    filepath = os.path.join(filepath,flag)
    filenames = os.listdir(filepath)  #读取nii文件
    slice_trans = []

    for f in filenames:
        #开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  #读取nii
        img_fdata = img.get_fdata()
        # fname = f.replace('.nii', '') #去掉nii的后缀名
        # img_f_path = os.path.join(newfilepath, fname)
        # 创建nii对应图像的文件夹
        # if not os.path.exists(img_f_path):
        #     os.mkdir(img_f_path)  #新建文件夹
        # if(flag=="image"):
        if flag == 'MR':
            img_fdata=(img_fdata-img_fdata.min())/(img_fdata.max()-img_fdata.min())*255
        #开始转换图像
        (x,y,z) = img.shape
        for i in range(z):   #是z的图象序列
            slice = img_fdata[:, :, i]  #选择哪个方向的切片自己决定
            # print(os.path.join(newfilepath, '{}_{}.png'.format(fname,i)))
            # cv2.imwrite(os.path.join(newfilepath, '{}_{}.png'.format(fname,i)), slice)
            slice[slice==9] =0
            slice[slice==10] =0
            slice = cv2.resize(slice, (448,224))
            save_path = os.path.join(newfilepath,flag+'_'+str(i),f.replace('nii', 'png'))
            cv2.imwrite(save_path, slice)



if __name__ == '__main__':
    oldfilepath = 'E:\\file\\start_data_5_22'#nii文件所在的文件夹路径
    newfilepath = 'E:\\file\\start_data_5_22\\image_440'#转化后的png文件存放的文件路径
    # nii_to_image(oldfilepath,newfilepath,"MR")
    nii_to_image(oldfilepath,newfilepath,"MR")
    nii_to_image(oldfilepath,newfilepath,"Mask")


#生成图片不分层的代码
# def nii_to_image(filepath,newfilepath,flag="image"):
#     filenames = os.listdir(filepath)  #读取nii文件
#     slice_trans = []

#     for f in filenames:
#         #开始读取nii文件
#         img_path = os.path.join(filepath, f)
#         img = nib.load(img_path)  #读取nii
#         img_fdata = img.get_fdata()
#         img_f_path =newfilepath
#         fname = f.replace('.nii', '') #去掉nii的后缀名
#         # img_f_path = os.path.join(newfilepath, fname)
#         # 创建nii对应图像的文件夹
#         # if not os.path.exists(img_f_path):
#         #     os.mkdir(img_f_path)  #新建文件夹
#         # if(flag=="image"):
#         img_fdata=(img_fdata-img_fdata.min())/(img_fdata.max()-img_fdata.min())*255
#         #开始转换图像
#         (x,y,z) = img.shape
#         for i in range(z):   #是z的图象序列
#             slice = img_fdata[:, :, i]  #选择哪个方向的切片自己决定
#             print(os.path.join(img_f_path, '{}_{}.png'.format(fname,i)))
#             cv2.imwrite(os.path.join(img_f_path, '{}_{}.png'.format(fname,i)), slice)

# if __name__ == '__main__':
#     oldfilepath = 'E:\\file\\start_data_5_22\\MR'#nii文件所在的文件夹路径
#     newfilepath = 'E:\\file\\start_data_5_22\\image\\MR'#转化后的png文件存放的文件路径
#     nii_to_image(oldfilepath,newfilepath,"image")