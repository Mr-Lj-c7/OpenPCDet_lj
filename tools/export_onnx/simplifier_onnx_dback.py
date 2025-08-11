# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    # op_attrs["dense_shape"] = np.array([496,432])  # [height, width]  kitti
    # modfy by lj 2025.06.04
    op_attrs["dense_shape"] = np.array([496,480])  # custom 特征图尺寸[y_size,x_size],x_size = (x_max-x_min)/grid_size, y_size = (y_max-y_min)/grid_size

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

# 获取指定层节点
def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
  return next_node

# 重新定义模型输入张量名称、输出张量，返回onnx模型, pointpillar_lj.yaml(沿用kitti配置)
def simplify_postprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)  # 将模型转换成一个伪图（Graph）对象

  # 根据配置文件中的类别数量动态计算输出通道数
  # 假设有4个类别（Car, Bus, Pedestrian, Motorcycle)
  num_classes = 3
  cls_preds_channels = num_classes * 3 * 2  # class_name.size * anchor_sizes * anchor_rotations.size = 4 * 3 * 2 = 24
  box_preds_channels = num_classes * 7 * 2   # 4个类别 * 7个回归参数 * 2个方向 = 56
  dir_cls_preds_channels = num_classes * 2 * 2  # 4个类别 * 2个方向 * 2个方向分数 = 16

  # 输出张量类别预测、包围框预测、方向预测，张量形状[batch_size, height, width, channels]
  # 
  '''
  cls_preds.shape = (1, 248, 240, cls_preds_channels),248, 240为下采样2倍后的特征图大小(496/2, 480/2),在yaml文件中LAYER_STRIDES: [2, 2, 2];
  cls_preds_channels = class_name.size * anchor_sizes * anchor_rotations.size = 4个类别 * 3个尺寸 * 2个方向; 
  box_preds_channels = 4个类别 * 7个回归参数(7参数:dx,dy,dz,dw,dl,dh,dyaw) * 2个方向;
  dir_cls_preds_channels = 4个类别 * 2个方向 * 2个方向分数;
  '''
  cls_preds = gs.Variable(name="cls_preds", dtype=np.float32, shape=(1, 248, 240, cls_preds_channels))
  box_preds = gs.Variable(name="box_preds", dtype=np.float32, shape=(1, 248, 240, box_preds_channels))
  dir_cls_preds = gs.Variable(name="dir_cls_preds", dtype=np.float32, shape=(1, 248, 240, dir_cls_preds_channels))

  tmap = graph.tensors()  # 获取所有张量
  # 输入张量voxels、voxel_idxs、voxel_num
  new_inputs = [tmap["voxels"], tmap["voxel_idxs"], tmap["voxel_num"]]
  # 输出张量
  new_outputs = [cls_preds, box_preds, dir_cls_preds]

  # 清除非new_inputs中的张量的输出连接
  for inp in graph.inputs:
    if inp not in new_inputs:
      inp.outputs.clear()

  # 清除所有张量的输入连接
  for out in graph.outputs:
    out.inputs.clear()

  # 找到第一个ConvTranspose节点
  first_ConvTranspose_node = [node for node in graph.nodes if node.op == "ConvTranspose"][0]
  # loop_node从当前节点向后查找第3层，找到Concat节点
  concat_node = loop_node(graph, first_ConvTranspose_node, 3)
  # 找到Concat节点
  assert concat_node.op == "Concat"

  # concat节点后的第一个节点
  first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]]

  # 循环修改Transpose节点后的输出为new_output输出张量
  for i in range(3):
    transpose_node = loop_node(graph, first_node_after_concat[i], 1)
    assert transpose_node.op == "Transpose"
    transpose_node.outputs = [new_outputs[i]]

  # 修改模型输入输出张量
  graph.inputs = new_inputs
  graph.outputs = new_outputs
  # 清理图并重新拓扑排序，确保图的结构合理且无冗余节点
  graph.cleanup().toposort()
  
  # 返回修改后的onnx模型
  return gs.export_onnx(graph)

# 重新定义模型输入张量、输出张量名称，返回onnx模型
def simplify_preprocess(onnx_model):
  print("Use onnx_graphsurgeon to modify onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  MAX_VOXELS = tmap["voxels"].shape[0]  # 获取最大体素数

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  # 输入张量voxels = [MAX_VOXELS, 32, 10],最大体素数、32为每个体素的点数、10为每个体素的特征数
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, 32, 10))

  # voxel_idxs: [V, 4]
  # V is the maximum number of voxels per frame
  # 4 is just the length of indexs encoded as (frame_id, z, y, x).
  # 输入张量voxel_idxs = [MAX_VOXELS, 4],最大体素数、每个体素的索引由四个值组成（帧ID、z、y、x）
  X = gs.Variable(name="voxel_idxs", dtype=np.int32, shape=(MAX_VOXELS, 4))

  # voxel_num: [1]
  # Gives valid voxels number for each frame
  # 输入张量voxel_num = [1],有效体素数量，1表示每一帧的有效体素数
  Y = gs.Variable(name="voxel_num", dtype=np.int32, shape=(1,))

  # 找到第一个Conv节点
  first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0]

  # 查找第一个MatMul节点
  first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0]

  # 找到MatMul节点后的第6个节点ReduceMax，设置keepdims为0，即不保留输入张量的维度
  next_node = current_node = first_node_pillarvfe
  for i in range(6):
    next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]
    if i == 5:              # ReduceMax
      current_node.attrs['keepdims'] = [0]
      break
    current_node = next_node

  last_node_pillarvfe = current_node

  #merge some layers into one layer between inputs and outputs as below
  graph.inputs.append(Y)  # 新定义的 Y 输入添加到图的输入列表
  # 重新定义输入[last_node_pillarvfe的输出， voxel_idxs，voxel_num]，输出
  inputs = [last_node_pillarvfe.outputs[0], X, Y]
  outputs = [first_node_after_pillarscatter.inputs[0]]
  # 将指定的子图替换为一个剪辑操作（Clip）
  graph.replace_with_clip(inputs, outputs)

  # Remove the now-dangling subgraph.
  # 清理图并重新拓扑排序，确保图的结构合理且无冗余节点
  graph.cleanup().toposort()

  #just keep some layers between inputs and outputs as below
  # 更新模型的输入张量、输出张量名称
  graph.inputs = [first_node_pillarvfe.inputs[0] , X, Y]
  graph.outputs = [tmap["cls_preds"], tmap["box_preds"], tmap["dir_cls_preds"]]

  graph.cleanup()  # 刷新模型拓扑排序

  #Rename the first tensor for the first layer 
  # 更新输入张量，将voxel输入连接到网络起始位置
  graph.inputs = [input_new, X, Y]
  first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
  first_add.inputs[0] = input_new

  # 再次清理图并重新拓扑排序，确保图的结构正确
  graph.cleanup().toposort()

  # 导出ONNX
  return gs.export_onnx(graph)

if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
