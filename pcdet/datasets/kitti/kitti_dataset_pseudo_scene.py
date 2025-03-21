import copy
import pickle
from collections import defaultdict

import torch
import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.augmentor.augmentor_utils import (
    random_flip_along_x, random_flip_along_y, global_rotation, global_scaling, global_scaling_with_roi_boxes,
    random_image_flip_horizontal, random_local_translation_along_x, random_local_translation_along_y,
    random_local_translation_along_z, local_scaling, local_rotation
)
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pathlib import Path

class KittiDataset_PseudoScene(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, premode=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, premode=premode
        )

        self.unlabeled_kitti_infos = []
        self.unlabeled_sample_id_list = []
        # assert len(self.kitti_infos) == len(self.sample_id_list)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))
        self.data_augmentor_ssl = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR_SSL, self.class_names, logger=self.logger,
            database_path=self.database_path,
        ) if self.training else None
        self.data_augmentor_ema = None
        self.pseudo_num =  self.dataset_cfg.get('PSEUDO_NUMBER', 5)

        self.repeat = self.dataset_cfg.get('REPEAT', 1)
        self.del_gt_boxes = self.dataset_cfg.get('DEL_GT_BOXES', True)
        self.ssl_mode = True

    def set_ssl_mode(self, mode):
        self.ssl_mode = mode

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI SSL dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))
    
    def get_item_single(self, info, is_ssl=False):
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            if not is_ssl:
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            else:
                gt_names = np.array(['None'])
                gt_boxes_camera = np.array([[0, 0, 0, 0, 0, 0, 0,  0]]).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)                

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict, is_ssl=is_ssl)

        data_dict['image_shape'] = img_shape
        data_dict['mask'] = 0 if is_ssl else 1
        return data_dict

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
        index = index % len(self.kitti_infos)
        info = copy.deepcopy(self.kitti_infos[index])

        data_dict_labeled = self.get_item_single(info)
        if self.training and self.ssl_mode:
            index_unlabeled = np.random.choice(len(self.unlabeled_sample_id_list), 1)[0]
            info_unlabeled = copy.deepcopy(self.unlabeled_kitti_infos[index_unlabeled])

            data_dict_unlabeled = self.get_item_single(info_unlabeled, is_ssl=True)
            use_ssl = np.random.randint(0,self.pseudo_num)
            if use_ssl == 0:
                return data_dict_labeled
            else:
                return data_dict_unlabeled
        else:
            return data_dict_labeled

    def prepare_data(self, data_dict, is_ssl=False):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_augmentor = self.data_augmentor if not is_ssl else self.data_augmentor_ssl
            data_dict = data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib
        data_dict = self.set_lidar_aug_matrix(data_dict)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            if selected.shape[0] > 0:
                data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        if self.del_gt_boxes and self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        # data_dict.pop('gt_names', None)
        data_dict.pop('old_indices', None)
        
        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        if self.training:
            return int(len(self.kitti_infos) * self.repeat)
        else:
            return len(self.kitti_infos)


if __name__ == '__main__':
    pass