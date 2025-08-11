"""
2024.03.21
author:alian
数据预处理操作
1.数据集分割
"""
import os
import random
import shutil
import numpy as np
 
 
def get_train_val_txt_kitti(src_path):
    """
    数据格式:KITTI
    # For KITTI Dataset
    └── KITTI_DATASET_ROOT
        ├── training    <-- 7481 train data
        |   ├── image_2 <-- for visualization
        |   ├── calib
        |   ├── label_2
        |   └── velodyne
        └── testing     <-- 7580 test data
            ├── image_2 <-- for visualization
            ├── calib
            └── velodyne
            
    src_path: KITTI_DATASET_ROOT kitti文件夹
    """
    # 1.自动生成数据集划分文件夹ImageSets
    set_path = "%s/ImageSets/"%src_path
    if os.path.exists(set_path):    # 如果文件存在
        shutil.rmtree(set_path)     # 清空原始数据
        os.makedirs(set_path)       # 重新创建
    else:
        os.makedirs(set_path)       # 自动新建文件夹
    
    # 2.训练样本分割  生成train.txt val.txt trainval.txt
    # train_list = os.listdir(os.path.join(src_path,'training','velodyne'))
    train_list = os.listdir(os.path.join(src_path,'training','points'))
    random.shuffle(train_list)     # 打乱顺序，随机采样
    # 设置训练和验证的比例
    train_p = 0.8
    # train_p = 0.9
 
    # 开始写入分割文件
    f_train = open(os.path.join(set_path, "train.txt"), 'w')
    f_val = open(os.path.join(set_path, "val.txt"), 'w')
    f_trainval = open(os.path.join(set_path, "trainval.txt"), 'w')
    
    for i,src in enumerate(train_list):
        if i<int(len(train_list)*train_p):    # 训练集的数量
            f_train.write(src[:-4] + '\n')
            f_trainval.write(src[:-4] + '\n')
        else:
            f_val.write(src[:-4] + '\n')
            f_trainval.write(src[:-4] + '\n')
 
    # 3.测试样本分割  生成test.txt
    # test_list = os.listdir(os.path.join(src_path,'testing','velodyne'))
    test_list = os.listdir(os.path.join(src_path,'testing','points'))
    f_test = open(os.path.join(set_path, "test.txt"), 'w')
    for i,src in enumerate(test_list):
        f_test.write(src[:-4] + '\n')
 
 
if __name__=='__main__':
    """
    	src_path: 数据目录
    """
    # src_path = '/home/dxy/Openpcdet-Test-master/data/custom'
    src_path = '/home/user/ChangTing/Code/Openpcdet/OpenPCDet/data/custom'
    get_train_val_txt_kitti(src_path)
