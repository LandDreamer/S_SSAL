import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_selected_dataloader, build_active_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils
import pickle
import random
from pcdet.models import load_data_to_gpu
from tqdm import tqdm
from ssl_utils.kitti_ssl_tools import (
    construct_kitti_ssl_gt_database_from_scenes, construct_kitti_ssl_pseudo_database_from_scenes, iter_reconstruct_kitti_ssl_pseudo_database_from_scenes,
)
from ssl_utils.waymo_ssl_tools import (
    construct_waymo_ssl_gt_database_from_scenes, construct_waymo_ssl_pseudo_database_from_scenes,
)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--pretrained_model', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--close_leave_bar', action='store_true', default=False, help='')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_2', type=str, default=None, help='extra tag for this experiment')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')

    parser.add_argument('--dist', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')    
    parser.add_argument('--construct_set', type=str, default='all', help='gt pseudo all')
    parser.add_argument('--random_seed', type=int, default=123, help='random_seed')
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

    
    ############### to do -> put al sample in outdir()
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    if args.extra_tag_2 is not None:
        output_dir = output_dir / args.extra_tag_2
    output_dir.mkdir(parents=True, exist_ok=True)

    dist = args.dist

    data_save_path = output_dir / ''
    log_file = output_dir / ('pretrain_construct_ssl_database_%s.txt' % 'pretrain')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # cfg.DATA_CONFIG.Dataset = 'KittiDataset'
    save_sample_path = None
    if 'ALSSL_TRAIN' in cfg and 'SAVE_SAMPLE_PATH' in cfg.ALSSL_TRAIN:
        save_sample_path = cfg.ALSSL_TRAIN.SAVE_SAMPLE_PATH
        
    labeled_set, unlabeled_set, labeled_loader, unlabeled_loader, \
        labeled_sampler, unlabeled_sampler = build_active_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1,
        dist, workers=args.workers, logger=logger, training=True,
        save_sample_path=save_sample_path,  shuffle=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=labeled_set)
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, logger=logger, to_cpu=dist)

    model.cuda()
    model.eval()
    if 'Kitti' in cfg.DATA_CONFIG.DATASET:
        if args.construct_set == 'all' or args.construct_set == 'gt':
            construct_kitti_ssl_gt_database_from_scenes(labeled_loader, output_dir, logger)
        if args.construct_set == 'all' or args.construct_set == 'pseudo':
            construct_kitti_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, output_dir, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False))
        if args.construct_set == 'iter':
            iter_reconstruct_kitti_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, output_dir, 2, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False))
    elif 'Waymo' in cfg.DATA_CONFIG.DATASET:
        if args.construct_set == 'all' or args.construct_set == 'gt':
            construct_waymo_ssl_gt_database_from_scenes(labeled_loader, output_dir, logger)
        if args.construct_set == 'all' or args.construct_set == 'pseudo':
            construct_waymo_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, output_dir, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False))
        if args.construct_set == 'iter':
            iter_reconstruct_waymo_ssl_pseudo_database_from_scenes(unlabeled_loader, model, cfg.ALSSL_TRAIN.SSL_Thresh, cfg.ALSSL_TRAIN.SSL_LOW_Thresh, output_dir, 2, logger, dynamic_thresh=cfg.ALSSL_TRAIN.get('DYNAMIC_TRHESH', False), keep_same_frame=False)

if __name__ == '__main__':
    main()



