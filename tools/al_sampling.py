import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import copy
import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu

from pcdet.datasets import build_active_dataloader, build_dataloader, build_selected_dataloader
import query_strategies
from ssl_utils.kitti_ssl_tools import (
    construct_kitti_ssl_gt_database_from_scenes, construct_kitti_ssl_pseudo_database_from_scenes,
)
from ssl_utils.waymo_ssl_tools import (
    construct_waymo_ssl_gt_database_from_scenes, construct_waymo_ssl_pseudo_database_from_scenes,
)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--al_round', type=int, default=1, help='specify al round')

    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--close_leave_bar', action='store_true', default=False, help='')

    parser.add_argument('--dist', action='store_true', default=False, help='')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_2', type=str, default='database', help='extra tag for this experiment')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--construct_set', type=str, default='gt', help='gt pseudo all')     
    parser.add_argument('--random_seed', type=int, default=666, help='random_seed')  
    parser.add_argument('--not_query', action='store_true', default=False, help='')  
    parser.add_argument('--change_pseudo', action='store_true', default=False, help='')  
    parser.add_argument('--init_construct_database', action='store_true', default=False, help='')  
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(args.random_seed + cfg.LOCAL_RANK)
    if args.change_pseudo and '_Pseudo' not in cfg.MODEL.NAME:
        cfg.MODEL.NAME = cfg.MODEL.NAME + '_Pseudo'
    if cfg.ALSSL_TRAIN.get('POST_PROCESSING_NMS_THRESH', None) is not None:
        if 'POST_PROCESSING' in cfg.MODEL and 'NMS_CONFIG' in cfg.MODEL.POST_PROCESSING and 'NMS_THRESH' in cfg.MODEL.POST_PROCESSING.NMS_CONFIG:
            cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = cfg.ALSSL_TRAIN.POST_PROCESSING_NMS_THRESH
    al_method = cfg.ALSSL_TRAIN.METHOD
    select_type = cfg.ALSSL_TRAIN.SELECT_TYPE
    rank = args.local_rank
    
    ############### to do -> put al sample in outdir()
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    database_dir = output_dir / args.extra_tag_2
    database_dir.mkdir(parents=True, exist_ok=True)

    dist = args.dist
    log_file = output_dir / ('log_al_round_%s.txt' % args.al_round)
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    log_config_to_file(cfg, logger=logger)
    logger.info(cfg.MODEL.NAME)
    
    labeled_set, unlabeled_set, labeled_loader, unlabeled_loader, \
        labeled_sampler, unlabeled_sampler = build_active_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1,
        dist, workers=args.workers, logger=logger, training=True,
        save_sample_path=cfg.ALSSL_TRAIN.get('INIT_SAMPLES_PATH', None), shuffle=False
    )
    if cfg.ALSSL_TRAIN.get('SELECTED_DATA_POOL', None) is not None:
        unlabeled_set, unlabeled_loader, unlabeled_sampler = build_selected_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist, workers=args.workers,
        logger=logger,
        training=True, test_training=True,
        seed=args.random_seed if args.fix_random_seed else None,
        save_sample_path_list=cfg.ALSSL_TRAIN.SELECTED_DATA_POOL, shuffle=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=unlabeled_set)
    model.cuda()
    
    model.load_params_from_file(filename=args.pretrained_model, logger=logger, to_cpu=dist)
    active_label_dir = output_dir / 'al_{}'.format(args.al_round)
    strategy = query_strategies.build_strategy(method=al_method, model=model, \
            labelled_loader=labeled_loader, \
            unlabelled_loader=unlabeled_loader, \
            rank=rank, \
            active_label_dir=active_label_dir, \
            cfg=cfg,
            logger=logger,
            model_list=None)
    save_path = []
    if not args.not_query:
        selected_frames = strategy.query(leave_pbar=(not args.close_leave_bar), al_round=args.al_round)
        save_path = strategy.save_active_labels(selected_frames=selected_frames, al_round=args.al_round)
        logger.info('successfully selecte frames ')
        logger.info('len %d' % len(selected_frames))
        logger.info(save_path)
        logger.info(selected_frames)
        save_path = [save_path]
    if cfg.ALSSL_TRAIN.get('INIT_SAMPLES_PATH', None):
        save_path.extend(cfg.ALSSL_TRAIN.INIT_SAMPLES_PATH)

    labeled_set, unlabeled_set, labeled_loader, unlabeled_loader, \
        labeled_sampler, unlabeled_sampler = build_active_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1,
        dist, workers=args.workers, logger=logger, training=False,
        save_sample_path=cfg.ALSSL_TRAIN.get('SAVE_SAMPLE_PATH',save_path), shuffle=False
    )
    if args.construct_set == 'all' or args.construct_set == 'gt':
        if 'Kitti' in cfg.DATA_CONFIG.DATASET:
            construct_kitti_ssl_gt_database_from_scenes(labeled_loader, database_dir, logger)
        elif 'Waymo' in cfg.DATA_CONFIG.DATASET:
            construct_waymo_ssl_gt_database_from_scenes(labeled_loader, database_dir, logger)
    if args.construct_set == 'all' or args.construct_set == 'pseudo':
        if 'Kitti' in cfg.DATA_CONFIG.DATASET:
            construct_kitti_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, database_dir, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False))
        elif 'Waymo' in cfg.DATA_CONFIG.DATASET:
            construct_waymo_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, database_dir, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False))

if __name__ == '__main__':
    main()