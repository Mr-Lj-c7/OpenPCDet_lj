import os
import json
import math
import numpy as np
import sys
 
def trans_detection_label(src_label_path, tgt_label_path, start_idx=None, end_idx=None):
    files = os.listdir(src_label_path)
    files.sort()  # 确保文件按名称排序
 
 
 
    # 初始化最大ID为0
    max_id = 0
 
    # 如果指定了 start_idx 和 end_idx，过滤出范围内的文件
    if start_idx is not None and end_idx is not None:
        files = [f for f in files if f.split('.')[0].isdigit() and start_idx <= int(f.split('.')[0]) <= end_idx]
 
    for fname in files:
        frame, _ = os.path.splitext(fname)
        print(frame)
 
        kitti_lines = []
        with open(os.path.join(src_label_path, fname), encoding='utf-8') as f:
            labels = json.load(f, strict=False)
            for label in labels:
                obj_type = label["obj_type"]
 
                # if label.get('obj_attr') == 'static':
                #     continue  # 跳过当前对象
                
                # 根据条件修改 obj_type
                if obj_type == 'Scooter':
                    obj_type = 'Bicycle'
                elif obj_type == 'Bus':
                    obj_type = 'Truck'
                
                if obj_type == 'Bicycle':
                    obj_type = 'Cyclist'
 
                box_id = int(label["obj_id"])
                box_id += 282
 
 
                # 更新最大ID
                if int(box_id) > max_id:
                    max_id = int(box_id)
 
 
                box_position_x = label['psr']['position']['x']
                box_position_y = label['psr']['position']['y']
                box_position_z = label['psr']['position']['z']
                box_scale_x = label['psr']['scale']['x']
                box_scale_y = label['psr']['scale']['y']
                box_scale_z = label['psr']['scale']['z']
                box_position_z_kitti = float(box_position_z) + 0 - float(box_scale_z / 2)
                rotation_yaw = -float(label['psr']['rotation']['z']) - math.pi / 2
 
                kitti_lines.append(f'{obj_type} 1.0 0 0.0 -1 -1 -1 -1 {box_scale_z:.4f} {box_scale_y:.4f} {box_scale_x:.4f} '
                                   f'{box_position_x:.4f} {box_position_y:.4f} {box_position_z_kitti:.4f} {rotation_yaw:.4f}\n')
 
            with open(os.path.join(tgt_label_path, frame + ".txt"), 'w') as outfile:
                outfile.writelines(kitti_lines)
                    # 在处理完所有文件后打印最大ID
    print(f"The maximum ID in the sequence is: {max_id}")
 
if __name__ == "__main__":
    src_label = "/home/dxy/SUSTechPOINTS/data/备份/label_copy"  # 替换成自己的路径
    tgt_label = "/home/dxy/SUSTechPOINTS/data/备份/label_kitti/"
    # 这里你可以指定开始和结束的索引，例如处理000001到001000范围内的文件
    start_idx = 1
    end_idx = 100
    trans_detection_label(src_label, tgt_label, start_idx, end_idx)
