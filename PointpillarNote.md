# PointPillar

## PointPillar -- 3D目标检测 
   概述：模型将三维点云数据转换为二维伪图像，基于二维卷积神经网络(CNN)对点云数据进行特征提取和编码，从而实现目标检测和定位。
   PointPillar 处理步骤：数据预处理（转换为俯视图进行特征增强） -- 利用PointNet进行特征提取 -- 检测头设计SSD（使用先验框进行目标检测） -- 损失计算
   
## PointPillars 模型网络结构
   1. Pillar Feature Net：将输入的点云转换为稀疏的伪图像的特征形式。
   2. Backbone（2D CNN）：使用 2D 的 CNN 处理伪图像特征得到高维度的特征。
   3. Detection Head（SSD）：检测和回归 3D 边界框。
   
## 三个损失函数Loss
   1. 类别损失（Focal Loss）
   2. 位置损失(Smooth L1 Loss)
   3. 方向分类损失(交叉熵损失)
   
## PointPillars模型推理
   1. 预处理，体素化特征提取 - Preprocess
   2. TRT推理 - Engine
   3. 预测结果后处理
   
## 优点
   1. 计算速度快：降低了点云数据的复杂度和计算量。
   2. 模型结构简单高效：具有非常高的检测精度和鲁棒性。
   3. 检测性能稳定：算法在各种不同场景下都具有稳定的检测性能。
   
   
## Pointpillars_Ros
   该工作空间下主要包括autoware_msgs、detected_objects_visualizer、lidar_point_pillars三个功能包，分别用于lidar感知结果消息编译、感知结果发布可视化、基于GPU的Pointpillars模型推理与障碍物检测，模型训练基于Openpcdet架构中的pointpillars网络，数据采用Kitti数据集验证测试。其中lidar_point_pillars主要分为检测和跟踪两部分结果。
