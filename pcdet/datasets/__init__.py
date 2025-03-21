import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
# from .once.once_dataset import ONCEDataset
# from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset

from .kitti.kitti_dataset_pseudo_scene import KittiDataset_PseudoScene
from .waymo.waymo_dataset_pseudo_scene import WaymoDataset_PseudoScene

from pcdet.config import cfg
import pickle
import random
from pathlib import Path

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    # 'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
    # 'Argo2Dataset': Argo2Dataset,

    'KittiDataset_PseudoScene': KittiDataset_PseudoScene,
    'WaymoDataset_PseudoScene': WaymoDataset_PseudoScene,
}


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, test_training=False,
                     shuffle=True):
    if not test_training:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
            premode='train'
        )    

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training and not test_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training and not test_training and shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    return dataset, dataloader, sampler

def build_selected_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                            logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0,
                            save_sample_path_list=[], shuffle=True, test_training=False, selected_frames_idx=[]):
    assert len(save_sample_path_list) + len(selected_frames_idx) > 0, 'load selected samples error'
    if not test_training:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )
        presample_set = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
            premode='train'
        )    
        presample_set = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
            premode='train'
        )

    if 'WaymoDataset' in cfg.DATA_CONFIG.DATASET:
        selected_frame_ids_list = selected_frames_idx
        for save_sample_path in save_sample_path_list:
            with open(save_sample_path, 'rb') as f:
                load_sample_id_list = pickle.load(f) 
            if 'frame_id' in load_sample_id_list:
                load_sample_id_list = load_sample_id_list['frame_id']
            selected_frame_ids_list.extend(load_sample_id_list)
        if len(selected_frame_ids_list) == 0:
            pass
        
        infos = dataset.infos
        pre_infos = []
        pre_frame_ids = []
        for info in infos:
            if info["frame_id"] in selected_frame_ids_list:
                pre_infos.append(info)
                pre_frame_ids.append(info["frame_id"])
        presample_set.infos = pre_infos
        presample_set.frame_ids = pre_frame_ids
        if logger is not None:
            logger.info('pre sample list')
            logger.info('len %d' % (len(presample_set.frame_ids)))
            logger.info(presample_set.frame_ids)
            logger.info('repeat len %d' % (len(presample_set)))

    else: # kitti case
        selected_sample_id_list = []
        for save_sample_path in save_sample_path_list:
            with open(save_sample_path, 'rb') as f:
                load_sample_id_list = pickle.load(f) 
            if 'frame_id' in load_sample_id_list:
                load_sample_id_list = load_sample_id_list['frame_id']
            assert len(load_sample_id_list) > 0, 'load presample idx error'
            selected_sample_id_list.extend(load_sample_id_list)

        if len(selected_sample_id_list) == 0:
            pass

        presample_set.sample_id_list = selected_sample_id_list
        presample_set.kitti_infos = [dataset.kitti_infos[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id in selected_sample_id_list]
        logger.info('pre sample list')
        logger.info('len %d' % (len(presample_set.sample_id_list)))
        logger.info(presample_set.sample_id_list)

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        presample_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler_labelled = torch.utils.data.distributed.DistributedSampler(presample_set)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler_labelled = DistributedSampler(presample_set, world_size, rank, shuffle=False)
    else:
        sampler_labelled, sampler_unlabelled =  None, None

    dataloader_labelled = DataLoader(
        presample_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_labelled is None) and training and shuffle, collate_fn=presample_set.collate_batch,
        drop_last=False, sampler=sampler_labelled, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
        )

    del dataset
    return presample_set, dataloader_labelled, sampler_labelled

def build_active_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                            logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0,
                            active_training=None, save_sample_path=None, shuffle=True):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=False,
        logger=logger,
        premode='train'
    )

    labelled_set = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=False,
        logger=logger,
        premode='train'
    )
    labelled_set.set_split('train')
    
    unlabelled_set = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=False,
        logger=logger,
        premode='train'
    )
    unlabelled_set.set_split('train')

    if active_training is not None:
        if 'Waymo' in cfg.DATA_CONFIG.DATASET:
            labelled_set.frame_ids, labelled_set.infos = \
                active_training[0], active_training[1]
            unlabelled_set.frame_ids, unlabelled_set.infos = \
                active_training[2], active_training[3]

        else: # kitti cases
            labelled_set.sample_id_list, labelled_set.kitti_infos = \
                active_training[0], active_training[1]
            unlabelled_set.sample_id_list, unlabelled_set.kitti_infos = \
                active_training[2], active_training[3]

    else:
        if 'Waymo' in cfg.DATA_CONFIG.DATASET:
            num_select_by_boxes = cfg.ALSSL_TRAIN.get('PRE_TRAIN_SAMPLE_BOX_NUMS', -1)
            selected_frame_ids_list = []
            for save_path in save_sample_path:
                with open(Path(save_path), 'rb') as f:
                    load_sample_id_list = pickle.load(f) 
                    if 'frame_id' in load_sample_id_list:
                        load_sample_id_list = load_sample_id_list['frame_id']
                    selected_frame_ids_list.extend(load_sample_id_list)
            assert len(selected_frame_ids_list) > 0, 'load presample idx error'
            labelled_set.frame_ids = selected_frame_ids_list
            unlabelled_set.frame_ids = [frame_id for frame_id in dataset.frame_ids if frame_id not in selected_frame_ids_list]
            labelled_set.infos = [dataset.infos[idx] for idx, frame_id in enumerate(dataset.frame_ids) if frame_id in selected_frame_ids_list]
            unlabelled_set.infos = [dataset.infos[idx] for idx, frame_id in enumerate(dataset.frame_ids) if frame_id not in selected_frame_ids_list]

        elif 'Kitti' in cfg.DATA_CONFIG.DATASET: # kitti case
            num_select_by_boxes = cfg.ALSSL_TRAIN.get('PRE_TRAIN_SAMPLE_BOX_NUMS', -1)
            selected_frame_ids_list = []
            for save_path in save_sample_path:
                with open(Path(save_path), 'rb') as f:
                    load_sample_id_list = pickle.load(f) 
                    if 'frame_id' in load_sample_id_list:
                        load_sample_id_list  = load_sample_id_list['frame_id']
                    selected_frame_ids_list.extend(load_sample_id_list)
            assert len(selected_frame_ids_list) > 0, 'load presample idx error'
            labelled_set.sample_id_list = selected_frame_ids_list
            unlabelled_set.sample_id_list = [sample_id for sample_id in dataset.sample_id_list if sample_id not in selected_frame_ids_list]
            labelled_set.kitti_infos = [dataset.kitti_infos[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id in selected_frame_ids_list]
            unlabelled_set.kitti_infos = [dataset.kitti_infos[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id not in selected_frame_ids_list]
            
            if logger is not None:
                logger.info('pre sample list')
                logger.info('len %d' % len(labelled_set.sample_id_list))
                logger.info(labelled_set.sample_id_list)

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        labelled_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
        unlabelled_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler_labelled = torch.utils.data.distributed.DistributedSampler(labelled_set)
            sampler_unlabelled = torch.utils.data.distributed.DistributedSampler(unlabelled_set)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler_labelled = DistributedSampler(labelled_set, world_size, rank, shuffle=False)
            sampler_unlabelled = DistributedSampler(unlabelled_set, world_size, rank, shuffle=False)
    else:
        sampler_labelled, sampler_unlabelled =  None, None


    dataloader_labelled = DataLoader(
        labelled_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_labelled is None) and training and shuffle, collate_fn=labelled_set.collate_batch,
        drop_last=False, sampler=sampler_labelled, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
        )
    dataloader_unlabelled = DataLoader(
        unlabelled_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_unlabelled is None) and training and shuffle, collate_fn=unlabelled_set.collate_batch,
        drop_last=False, sampler=sampler_unlabelled, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
        )

    del dataset
    return labelled_set, unlabelled_set, \
           dataloader_labelled, dataloader_unlabelled, \
           sampler_labelled, sampler_unlabelled


def build_ssl_dataloader_naive(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                            logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0,
                            save_sample_path_list=[], save_pre_sample=False, use_pre_sample=False, test_training=False, shuffle=True):
    assert len(save_sample_path_list) > 0, 'load selected samples error'
    if not test_training:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        ssl_set = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=True,
            logger=logger,
        )
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
            premode='train',
        )

        ssl_set = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
            premode='train',
        )        
    # Build ssl-train datasets and dataloaders before active training loop
    if 'WaymoDataset' in dataset_cfg.DATASET:
        selected_frame_ids_list = []
        for save_sample_path in save_sample_path_list:
            with open(save_sample_path, 'rb') as f:
                load_sample_id_list = pickle.load(f) 
            if 'frame_id' in load_sample_id_list:
                load_sample_id_list = load_sample_id_list['frame_id']                
            selected_frame_ids_list.extend(load_sample_id_list)
            
        ######## to do -> random select and save
        if len(selected_frame_ids_list) == 0:
            pass

        infos = dataset.infos
        labeled_infos = []
        labeled_frame_ids = []
        unlabeled_infos = []
        unlabeled_frame_ids = []

        for info in infos:
            if info["frame_id"] in selected_frame_ids_list:
                labeled_infos.append(info)
                labeled_frame_ids.append(info["frame_id"])
            else:
                unlabeled_infos.append(info)
                unlabeled_frame_ids.append(info["frame_id"])                
        ssl_set.infos = labeled_infos
        ssl_set.frame_ids = labeled_frame_ids
        ssl_set.unlabeled_infos = unlabeled_infos
        ssl_set.unlabeled_frame_ids = unlabeled_frame_ids       

        logger.info('ssl labeled sample list')
        logger.info('len %d' % (len(ssl_set.frame_ids)))
        logger.info(ssl_set.frame_ids)
        logger.info('ssl unlabeled sample list')
        logger.info('len %d' % (len(ssl_set.unlabeled_frame_ids)))
        # logger.info(ssl_set.unlabeled_frame_ids)

    elif 'KittiDataset' in dataset_cfg.DATASET: # kitti case
        selected_sample_id_list = []
        for save_sample_path in save_sample_path_list:
            with open(save_sample_path, 'rb') as f:
                load_sample_id_list = pickle.load(f) 
            if 'selected_frames_epoch' in save_sample_path:
                load_sample_id_list = load_sample_id_list['frame_id']
            assert len(load_sample_id_list) > 0, 'load presample idx error'
            selected_sample_id_list.extend(load_sample_id_list)

        if len(selected_sample_id_list) == 0:
            pass

        ssl_set.sample_id_list = selected_sample_id_list
        ssl_set.kitti_infos = [dataset.kitti_infos[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id in selected_sample_id_list]
        ssl_set.unlabeled_sample_id_list = [dataset.sample_id_list[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id not in selected_sample_id_list]
        ssl_set.unlabeled_kitti_infos = [dataset.kitti_infos[idx] for idx, sample_id in enumerate(dataset.sample_id_list) if sample_id not in selected_sample_id_list]
         

        logger.info('ssl labeled sample list')
        logger.info('len %d' % (len(ssl_set.sample_id_list)))
        logger.info(ssl_set.sample_id_list)
        logger.info('ssl unlabeled sample list')
        logger.info('len %d' % (len(ssl_set.unlabeled_sample_id_list)))
        # logger.info(ssl_set.unlabeled_sample_id_list)
    else:
        raise NotImplementedError

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        ssl_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler_ssl = torch.utils.data.distributed.DistributedSampler(ssl_set)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler_ssl = DistributedSampler(ssl_set, world_size, rank, shuffle=False)
    else:
        sampler_ssl, sampler_unlabelled =  None, None


    dataloader_ssl = DataLoader(
        ssl_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_ssl is None) and training and shuffle, collate_fn=ssl_set.collate_batch,
        drop_last=False, sampler=sampler_ssl, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
        )

    del dataset
    return ssl_set, dataloader_ssl, sampler_ssl



def select_init_frames(dataset_cfg, class_names, root_path=None, logger=None, save_sample_path=None):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=False,
        logger=logger,
        premode='train'
    )

    if 'Waymo' in cfg.DATA_CONFIG.DATASET:
        num_select_by_boxes = cfg.ALSSL_TRAIN.get('PRE_TRAIN_SAMPLE_BOX_NUMS', -1)
        if num_select_by_boxes > 0:
            infos = dataset.infos
            random.shuffle(infos)

            pre_infos = []
            pre_frame_ids = []
            selected_sum = 0
            s_idx = 0
            for idx, info in enumerate(infos):
                gt_boxes_names = info['annos']['name']
                num_boxes = len([gt_name for gt_name in gt_boxes_names if gt_name in class_names])
                
                selected_sum += num_boxes
                if selected_sum > num_select_by_boxes:
                    break
                s_idx = idx
                pre_infos.append(info)
                pre_frame_ids.append(info["frame_id"])

            with open(Path(save_sample_path[0]), 'wb') as f:
                pickle.dump(pre_frame_ids, f)
        else:
            infos = dataset.infos
            random.shuffle(infos)
            infos = infos[:cfg.ALSSL_TRAIN.PRE_TRAIN_SAMPLE_NUMS]
            pre_frame_ids = []
            for info in infos:
                pre_frame_ids.append(info["frame_id"])

            with open(Path(save_sample_path[0]), 'wb') as f:
                pickle.dump(pre_frame_ids, f)
        
    elif 'Kitti' in cfg.DATA_CONFIG.DATASET: # kitti case
        num_select_by_boxes = cfg.ALSSL_TRAIN.get('PRE_TRAIN_SAMPLE_BOX_NUMS', -1)
        if num_select_by_boxes > 0:
            pairs = list(zip(dataset.sample_id_list, dataset.kitti_infos))
            random.shuffle(pairs)
            sample_id_list, kitti_infos = \
                zip(*pairs)

            selected_sum = 0
            s_idx = 0
            for idx, info in enumerate(kitti_infos):
                gt_boxes_names = info['annos']['name']
                num_boxes = len([gt_name for gt_name in gt_boxes_names if gt_name in class_names])
                
                selected_sum += num_boxes
                if selected_sum > num_select_by_boxes:
                    break
                s_idx = idx
            sample_id_list, kitti_infos = \
                zip(*pairs[:s_idx+1])
            
            with open(save_sample_path[0], 'wb') as f:
                pickle.dump(sample_id_list, f)
        else:
            pairs = list(zip(dataset.sample_id_list, dataset.kitti_infos))
            random.shuffle(pairs)
            # labelled_set, unlabelled_set = copy.deepcopy(dataset), copy.deepcopy(dataset)
            sample_id_list, kitti_infos = \
                zip(*pairs[:cfg.ALSSL_TRAIN.PRE_TRAIN_SAMPLE_NUMS])
            
            with open(Path(save_sample_path[0]), 'wb') as f:
                pickle.dump(sample_id_list, f)