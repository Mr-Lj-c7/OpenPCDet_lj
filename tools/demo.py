import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# 导入box_colormap
if OPEN3D_FLAG:
    from visual_utils.open3d_vis_utils import box_colormap
else:
    from visual_utils.visualize_utils import box_colormap

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # modfy by lj 2025.08.05
# ----------------------------------------------------custom------------------------------------------------------------------
            # 查看推理结果的数量
            num_predictions = len(pred_dicts[0]['pred_boxes'])
            logger.info(f'检测到的对象数量: \t{num_predictions}')

            # 安全地访问预测结果进行可视化
            if pred_dicts and len(pred_dicts) > 0 and 'pred_boxes' in pred_dicts[0]:
                logger.info("正在可视化预测结果")
                # 获取预测框、分数和标签
                ref_boxes = pred_dicts[0]['pred_boxes']
                ref_scores = pred_dicts[0]['pred_scores']
                ref_labels = pred_dicts[0]['pred_labels']
                
                # 打印调试信息
                logger.info(f'预测框形状: {ref_boxes.shape if ref_boxes is not None else "None"}')
                logger.info(f'预测分数形状: {ref_scores.shape if ref_scores is not None else "None"}')
                logger.info(f'预测标签形状: {ref_labels.shape if ref_labels is not None else "None"}')
                if ref_labels is not None and len(ref_labels) > 0:
                    logger.info(f'预测标签值: {ref_labels}')
                # 调整检测框角度以匹配LiDAR坐标系
                if ref_boxes is not None and len(ref_boxes) > 0:
                    # 创建检测框的副本以避免修改原始数据
                    adjusted_boxes = ref_boxes.clone() if isinstance(ref_boxes, torch.Tensor) else ref_boxes.copy()
                    # 修正角度 - 根据KITTI数据集的坐标转换要求进行反向补偿
                    # KITTI格式转换: rotation_y = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    # 因此反向转换应该是: gt_boxes_lidar[:, 6] = -rotation_y - np.pi / 2.0
                    adjusted_boxes[:, 6] = -ref_boxes[:, 6] - np.pi / 2.0
                # 处理标签值，确保在有效范围内
                if ref_labels is not None and len(ref_labels) > 0:
                    # 将标签值减1，使其从0开始，与box_colormap索引对应
                    # OpenPCDet模型输出的标签从1开始，但可视化需要从0开始
                    adjusted_labels = ref_labels - 1
                    
                    # 确保标签值在有效范围内[0, len(box_colormap)-1]
                    adjusted_labels = torch.clamp(adjusted_labels, min=0, max=len(box_colormap)-1)
                    
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], 
                        # ref_boxes=ref_boxes,
                        ref_boxes=adjusted_boxes if 'adjusted_boxes' in locals() else ref_boxes,
                        ref_scores=ref_scores, 
                        ref_labels=adjusted_labels
                    )
                else:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], 
                        ref_boxes=ref_boxes,
                        ref_scores=ref_scores
                    )
            else:
                # 如果没有预测结果，只显示点云
                logger.info("未检测到任何对象，仅显示点云")
                V.draw_scenes(points=data_dict['points'][:, 1:])
# ----------------------------------------------------------------------------------------------------------------------
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
