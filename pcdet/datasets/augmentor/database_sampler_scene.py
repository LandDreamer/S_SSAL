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
import random

def random_delete_data(data_list, percentage=0.1):
    num_items = len(data_list)
    num_items_to_delete = int(num_items * percentage)

    indices_to_delete = random.sample(range(num_items), num_items_to_delete)
    modified_list = [data for i, data in enumerate(data_list) if i not in indices_to_delete]

    return modified_list

class DataBaseSampler_Scene(DataBaseSampler):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None, database_path=None):
        self.root_path = database_path
        self.database_path = database_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        self.logger = logger
        self.db_infos = {}
        self.scene_db_infos = []
        self.frame_db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            self.db_infos['BACK_' + class_name] = []
        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = Path(db_info_path)
            if not db_info_path.exists():
                logger.info('Not load {} info'.format(db_info_path))
                continue
            if logger is not None:
                logger.info('load {} info'.format(db_info_path))
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos['box'][cur_class]) for cur_class in class_names if cur_class in infos['box']]
                [self.db_infos['BACK_' + cur_class].extend(infos['box']['BACK_' + cur_class]) for cur_class in class_names if 'BACK_' + cur_class in infos['box']]

        self.drop_box_class_rate = sampler_cfg.get('DROP_BOX_CLASS_RATE', 0)
        for scene_db_info_path in sampler_cfg.SCENE_DB_INFO_PATH:
            scene_db_info_path = Path(scene_db_info_path)
            if not scene_db_info_path.exists():
                logger.info('Not load {} info'.format(scene_db_info_path))
                continue
            if logger is not None:
                logger.info('load {} info'.format(scene_db_info_path))
            with open(str(scene_db_info_path), 'rb') as f:
                infos = pickle.load(f)
                self.scene_db_infos.extend(infos['scene'])


        self.is_fill_frame = False
        if 'FRAME_DB_INFO_PATH' in sampler_cfg:
            for frame_db_info_path in sampler_cfg.FRAME_DB_INFO_PATH:
                self.is_fill_frame = True
                frame_db_info_path = Path(frame_db_info_path)
                if not frame_db_info_path.exists():
                    logger.info('Not load {} info'.format(frame_db_info_path))
                    continue
                if logger is not None:
                    logger.info('load {} info'.format(frame_db_info_path))
                with open(str(frame_db_info_path), 'rb') as f:
                    infos = pickle.load(f)
                    self.frame_db_infos.update(infos)

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

        self.not_shuffle = sampler_cfg.get('Not_Shuffle', False)
        self.index_scene_ = 0 

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            if self.not_shuffle:
                indices = np.array([i for i in range(len(self.db_infos[class_name]))])
            else:
                indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sample_group['old_pointer'] = pointer
        sample_group['old_indices'] = indices

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        if self.not_shuffle:
            index_scene = self.index_scene_ % len(self.scene_db_infos)
            self.index_scene_ = self.index_scene_ + 1
        else:
            index_scene = np.random.choice(len(self.scene_db_infos), 1)[0]

        points = np.fromfile(str(self.scene_db_infos[index_scene]['path']), dtype=np.float32).reshape(
            [-1, self.sampler_cfg.NUM_POINT_FEATURES])
        scene_id = self.scene_db_infos[index_scene]['sample_idx']
        data_dict['frame_id'] = scene_id
        data_dict['points'] = points
        data_dict['gt_boxes'] = []
        data_dict['gt_names'] = []
        
        existed_boxes = None
        total_valid_sampled_dict = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []
        sample_indices = {}
        frame_box_info = {}
        if self.is_fill_frame and scene_id in self.frame_db_infos:
            frame_box_info = self.frame_db_infos[scene_id]
        for class_name, sample_group in self.sample_groups.items():
            if self.is_fill_frame and class_name in frame_box_info:
                sample_exit_boxes = [boxes for boxes in frame_box_info[class_name] if boxes['num_points_in_gt'] > 5]
            else:
                sample_exit_boxes = []
            num_gt = len(sample_exit_boxes)
            if self.limit_whole_scene:
                sample_group['sample_num'] = str(max(int(self.sample_class_num[class_name]) - num_gt, 0))
            if int(self.sample_class_num[class_name]) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                sample_indices[class_name] = {'pointer': sample_group['old_pointer'], 'indices': sample_group['old_indices']}

                sampled_dict = sample_exit_boxes + sampled_dict
                # if len(sampled_dict) == 0:
                #     continue
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)
                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'
                if existed_boxes is not None:
                    iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                    iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                    iou2[iou2 < 0.3] = 0
                    iou1 = iou1 if iou1.shape[1] > 0 else iou2
                    valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                    if self.img_aug_type is not None:
                        raise NotImplementedError
                        sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask)
                        sampled_gt_boxes2d.append(sampled_boxes2d)
                        if mv_height is not None:
                            sampled_mv_height.append(mv_height)

                    valid_mask = valid_mask.nonzero()[0]
                    valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                    valid_sampled_boxes = sampled_boxes[valid_mask]
                else:
                    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                    iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                    iou2[iou2 < 0.3] = 0
                    valid_mask = ((iou2.max(axis=1)) == 0)
                    valid_mask = valid_mask.nonzero()[0]
                    valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                    valid_sampled_boxes = sampled_boxes[valid_mask]
                if existed_boxes is None:
                     existed_boxes = valid_sampled_boxes
                else:
                    existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes
        data_dict['old_indices'] = sample_indices

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )
        data_dict.pop('gt_boxes_mask')
        return data_dict
    
    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        points = data_dict['points']    
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )

        obj_points_list = []


        gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            assert obj_points.shape[0] == info['num_points_in_gt'], "error path %s, %d != %d" % (file_path, obj_points.shape[0], info['num_points_in_gt'])
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                raise NotImplementedError
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        new_obj_points_list = obj_points_list
        obj_points = np.concatenate(new_obj_points_list, axis=0)
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)

        keep_indices = self.cer_class_indices(sampled_gt_names, 'BACK')
        sampled_gt_names = sampled_gt_names[keep_indices]
        sampled_gt_boxes = sampled_gt_boxes[keep_indices]
        
    
        gt_names = sampled_gt_names
        gt_boxes = sampled_gt_boxes
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:
            raise NotImplementedError
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)
        return data_dict