# Pointpillar params 

## 
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist'] - 要检测的目标类别
   

## 
""" 
_BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml - 基于KITTI数据集
POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1] - 激光雷达检测范围阈值
"""

## 
“”“
NAME: mask_points_and_boxes_outside_range - 移除检测范围外的点云和框，只处理范围内的点
REMOVE_OUTSIDE_BOXES: True
”“”

##
“”“
NAME: shuffle_points - 在训练过程中打乱点的顺序，增强泛化能力。测试时不打乱。
SHUFFLE_ENABLED: {'train': True, 'test': False}
”“”

##
“”“
NAME: transform_points_to_voxels - 将点云数据转化为体素数据
VOXEL_SIZE: [0.16, 0.16, 4] - 体素的大小为 [0.16, 0.16, 4] 米
MAX_POINTS_PER_VOXEL: 32 - 每个体素最多32个点
“”“

##
“”“
AUG_CONFIG_LIST:
    - NAME: gt_sampling - 随机采样点云数据进行数据增强
      SAMPLE_GROUPS: ['Car:15','Pedestrian:15', 'Cyclist:15'] - 汽车、行人、骑行者，样本数为15
“”“

## 
“”“
NAME: PointPillar - 模型名称
”“”

##
“”“
NAME: PillarVFE - 点云的体素编码器，生成64维特征，点云数据序列化
WITH_DISTANCE: False - 不使用距离信息
NUM_FILTERS: [64] - 64维特征
“”“

##
“”“
NUM_BEV_FEATURES: 64 - 点云特征映射为鸟瞰视图（BEV）特征
“”“

## 2D卷积网络的主干部分
“”“
LAYER_NUMS: [3, 5, 5] -  3个2D卷积层，5个2D卷积， 5个2D卷积层
NUM_FILTERS: [64, 128, 256] - 每层卷积的过滤器数量分别为64, 128和256
UPSAMPLE_STRIDES: [1, 2, 4] - 上采样步长[1, 2, 4]
“”“

## NAME: AnchorHeadSingle - 检测头，用于生成锚框（anchors）
"""
USE_DIRECTION_CLASSIFIER: True - 设置方向分类器（Direction Classifier）
DIR_OFFSET: 0.78539
NUM_DIR_BINS: 2 - 有2个方向（0度和90度）
"""

##
"""
’clss_name‘: 'Car'
‘anchor_sizes’: [[3.9, 1.6, 1.56]] - 锚框的尺寸,汽车的大小为 3.9米 × 1.6米 × 1.56米
'anchor_rotations': [0, 1.57], - 锚框的旋转角度，汽车为0度和90度
'anchor_bottom_heights': [-1.78], - 锚框底部高度，相对于地面或传感器坐标系的偏移值
'align_center': False, - 不使用中心点对齐，为了更好地处理目标在场景中的分布和检测
'feature_map_stride': 2, - 特征图上的步幅大小，决定了特征图的分辨率和锚框的密集程度
'matched_threshold': 0.6, - 匹配阈值，用于确定锚框是否与检测框匹配，正样本将用于回归定位和分类
'unmatched_threshold': 0.45 - 未匹配阈值，用于确定锚框是否与检测框匹配，负样本将用于训练
"""

## LOSS_CONFIG - 损失函数中不同部分的权重
"""
LOSS_WEIGHTS: {
    'cls_weight': 1.0, - 分类损失权重为1.0
    'loc_weight': 2.0, - 位置损失权重为2.0
    'dir_weight': 0.2, - 方向损失权重为0.2
}
"""

## POST_PROCESSING - 去除重叠的检测框
"""
NMS_CONFIG:
    NMS_TYPE: nms_gpu - 非极大值抑制（NMS）的类型为GPU加速版本
    NMS_THRESH: 0.01 - NMS阈值
"""

## OPTIMIZATION
"""
BATCH_SIZE_PER_GPU: 4 - 每个GPU的批量大小为4
NUM_EPOCHS: 80 - 训练轮数为80
OPTIMIZER: adam_onecycle - 使用Adam优化器，并采用OneCycle学习率策略
LR: 0.003 - 初始学习率为0.003
WEIGHT_DECAY: 0.01
"""
