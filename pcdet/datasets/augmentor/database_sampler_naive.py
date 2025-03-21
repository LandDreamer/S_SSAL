import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
# import SharedArray
import torch.distributed as dist
from pathlib import Path
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common
from pcdet.datasets.augmentor.database_sampler import DataBaseSampler

class DataBaseSampler_Naive(DataBaseSampler):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None, database_path=None):
        self.root_path = database_path
        self.database_path = database_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            self.db_infos['BACK_' + class_name] = []

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            # db_info_path = self.database_path.resolve() / db_info_path
            db_info_path = Path(db_info_path)
            if not db_info_path.exists():
                if logger is not None:
                    logger.info('Not load {} info'.format(db_info_path))
                continue
            if logger is not None:
                logger.info('load {} info'.format(db_info_path))
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos['box'][cur_class]) for cur_class in class_names if cur_class in infos['box']]
                [self.db_infos['BACK_' + cur_class].extend(infos['box']['BACK_' + cur_class]) for cur_class in class_names if 'BACK_' + cur_class in infos['box']]
                    

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
            

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                # 'pointer': -1,
                'indices': np.arange(len(self.db_infos[class_name]))
            }
        if 'BACK_SAMPLE_GROUPS' in sampler_cfg: 
            for x in sampler_cfg.BACK_SAMPLE_GROUPS:
                class_name, sample_num = x.split(':')
                if class_name not in class_names:
                    continue
                self.sample_class_num['BACK_' + class_name] = sample_num
                self.sample_groups['BACK_' + class_name] = {
                    'sample_num': sample_num,
                    'pointer': len(self.db_infos['BACK_' + class_name]),
                    # 'pointer': -1,
                    'indices': np.arange(len(self.db_infos['BACK_' + class_name]))
                }

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            # data_dict.pop('calib')
            # data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                # file_path = self.root_path / info['path']
                file_path = info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)
            if obj_points.shape[0] != info['num_points_in_gt']:
                print(file_path)
            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)


        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict