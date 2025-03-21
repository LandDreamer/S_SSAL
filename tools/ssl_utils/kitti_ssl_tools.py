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
import pickle
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_selected_dataloader, build_active_dataloader
from pcdet.models import build_network
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from pcdet.models import load_data_to_gpu
from pcdet.utils import box_utils, common_utils, calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from sklearn.cluster import KMeans

def construct_kitti_ssl_gt_database_from_scenes(labeled_dataloader, save_dir, logger):
    class_names = labeled_dataloader.dataset.class_names
    dataset_name = cfg.DATA_CONFIG.DATASET.strip('Dataset')
    assert 'Kitti' in dataset_name, "not Kitti dataset "
    box_database_save_path = Path(save_dir) / ('ssl_gt_database_box')
    scene_database_save_path = Path(save_dir) / ('ssl_gt_database_scene')
    db_info_save_path = Path(save_dir) / ('ssl_dbinfos_gt.pkl')
    frame_info_save_path = Path(save_dir) / ('frame_dbinfos_gt.pkl')

    box_database_save_path.mkdir(parents=True, exist_ok=True)
    scene_database_save_path.mkdir(parents=True, exist_ok=True) 

    all_db_infos = {'scene':[], 'box':{}}

    labeled_dataset = labeled_dataloader.dataset
    labeled_infos = labeled_dataset.kitti_infos
    frame_dict = {}

    for k in tqdm(range(len(labeled_infos))):
        info = labeled_infos[k]
        sample_idx = info['point_cloud']['lidar_idx']
        points = labeled_dataset.get_lidar(sample_idx)
        calib = labeled_dataset.get_calib(sample_idx)
        img_shape = info['image']['image_shape']
        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        bbox = annos['bbox'] #(5, 7)
        gt_boxes = annos['gt_boxes_lidar']

        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = labeled_dataset.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  

        points_no_obj = box_utils.remove_points_in_boxes3d(points, gt_boxes)
        scene_path = scene_database_save_path / 'scenes_no_object_{}_{}.bin'.format(k, sample_idx)
        sample_idx = int(sample_idx)
        with open(scene_path, 'w') as f:
            points_no_obj.tofile(f) 
        scene_info = {'sample_idx': sample_idx, 'path': scene_path}   
        info_record = {'scene': [len(all_db_infos['scene'])]}
        for cl in class_names: 
            info_record[cl] = []   
            info_record['BACK_'+cl] = [] 
        all_db_infos['scene'].append(scene_info)

        for i in range(num_obj):
            if names[i] not in class_names:
                continue
            filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
            filepath = box_database_save_path / filename
            gt_points = points[point_indices[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
            
            # db_path = str(filepath.relative_to(save_dir))  # gt_database/xxxxx.bin
            db_path = filepath
            db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i],
                        'flag': 'GT', 'round': 0, 'iter_time':0}

            if names[i] in all_db_infos['box']:
                all_db_infos['box'][names[i]].append(db_info)
            else:
                all_db_infos['box'][names[i]] = [db_info]

            
            info_record[names[i]].append(len(all_db_infos['box'][names[i]]) - 1)

        frame_dict[sample_idx] = info_record
    logger.info('Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict


def get_kitti_ssl_pseudo_dynamic_thresh(unlabeled_datalodar, model, save_dir, logger, save_preds=True, right_center_num=10, wrong_center_num=10):
    logger.info('---------- Get thresh -------------------')
    frame_pred_result_save_path = Path(save_dir) / ('model_last_pred_res.pkl')
    model.eval()
    val_dataloader_iter = iter(unlabeled_datalodar)
    val_loader = unlabeled_datalodar
    total_it_each_epoch = len(unlabeled_datalodar) 
    frame_pred_res = {}
    all_boxes_pseudo = {cl+1: [] for cl in range(len(unlabeled_datalodar.dataset.class_names))}

    for cur_it in tqdm(range(total_it_each_epoch)):
        # if cur_it > 300:
        #     break
        try:
            unlabeled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabeled_dataloader_iter = iter(val_loader)
            unlabeled_batch = next(unlabeled_dataloader_iter)
        
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                sample_idx = unlabeled_batch['frame_id'].item()
                pred_boxes_loc = pred_dicts[batch_inx]['pred_boxes'].cpu().numpy() #(15 or ..., 7)
                
                # if pred_boxes_loc.numel() == 0:
                #     continue
                pred_labels = pred_dicts[batch_inx]['pred_labels'].cpu().numpy() #(15 or ...)
                pred_scores = pred_dicts[batch_inx]['pred_scores'].cpu().numpy() #(15 or ...)
                frame_info = {
                    'pred_boxes': pred_boxes_loc,
                    'pred_labels': pred_labels,
                    'pred_scores': pred_scores
                }
                frame_pred_res[sample_idx] = frame_info
                pred_num = pred_labels.shape[0]
                for p_id in range(pred_num):
                    all_boxes_pseudo[pred_labels[p_id]].append(pred_scores[p_id])
    if save_preds:
        with open(frame_pred_result_save_path, 'wb') as f:
            pickle.dump(frame_pred_res, f)   
    thresh = []
    low_thresh = []
    for c_id, cl in enumerate(unlabeled_datalodar.dataset.class_names):
        X = all_boxes_pseudo[c_id+1]
        if type(right_center_num) == list:
            k1 = right_center_num[c_id]
        else:
            k1 = right_center_num
        if type(wrong_center_num) == list:
            k2 = wrong_center_num[c_id]
        else:
            k2 = wrong_center_num
        k = min(k1+k2, len(X))
        if k > 0:
            kmeans = KMeans(n_clusters=k)
            X = np.array(X).reshape(-1, 1)
            kmeans.fit(X)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            centroids = [center.item() for center in sorted(centroids)]
            logger.info('-------------------------')
            logger.info(f'{cl} all pseudo')
            logger.info('centers')
            logger.info(centroids)
            thresh.append(centroids[-1])
            low_thresh.append(centroids[0])
    return thresh, low_thresh


def get_kitti_ssl_pseudo_dynamic_thresh_iter(unlabeled_datalodar, model, save_dir, logger, save_preds=True, right_center_num=10, wrong_center_num=10):
    logger.info('---------- Get iter thresh -------------------')
    frame_pred_result_save_path = Path(save_dir) / ('model_last_pred_res.pkl')
    model.eval()
    val_dataloader_iter = iter(unlabeled_datalodar)
    val_loader = unlabeled_datalodar
    total_it_each_epoch = len(unlabeled_datalodar) 
    new_frame_pred_res = {}
    frame_pred_res = {}
    class_names = unlabeled_datalodar.dataset.class_names
    # all_boxes_pseudo = {cl+1: [] for cl in range(len(unlabeled_datalodar.dataset.class_names))}
    right_boxes_pseudo = {cl+1: [] for cl in range(len(class_names))}
    wrong_boxes_pseudo = {cl+1: [] for cl in range(len(class_names))}
    with open(frame_pred_result_save_path, 'rb') as f:
        frame_pred_res = pickle.load(f)  

    for cur_it in tqdm(range(total_it_each_epoch)):
        # if cur_it > 300:
        #     break
        try:
            unlabeled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabeled_dataloader_iter = iter(val_loader)
            unlabeled_batch = next(unlabeled_dataloader_iter)
        
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                sample_idx = unlabeled_batch['frame_id'].item()
                pred_boxes_loc = pred_dicts[batch_inx]['pred_boxes'].cpu().numpy() #(15 or ..., 7)
                
                # if pred_boxes_loc.numel() == 0:
                #     continue
                pred_labels = pred_dicts[batch_inx]['pred_labels'].cpu().numpy() #(15 or ...)
                pred_scores = pred_dicts[batch_inx]['pred_scores'].cpu().numpy() #(15 or ...)
                new_frame_info = {
                    'pred_boxes': pred_boxes_loc,
                    'pred_labels': pred_labels,
                    'pred_scores': pred_scores
                }
                new_frame_pred_res[sample_idx] = new_frame_info

                frame_info = frame_pred_res[sample_idx]
                if pred_boxes_loc.shape[0] == 0:
                    continue
                old_pred_boxes_loc = frame_info['pred_boxes']
                has_no_old = False
                if old_pred_boxes_loc.shape[0] == 0:
                    has_no_old = True
                else:
                    iou = iou3d_nms_utils.boxes_bev_iou_cpu(pred_boxes_loc[:, 0:7], old_pred_boxes_loc[:, 0:7])
                
                pred_num = pred_labels.shape[0]
                for p_id in range(pred_num):
                    if not has_no_old and iou[p_id].max() > 0.5:
                        right_boxes_pseudo[pred_labels[p_id]].append(pred_scores[p_id])
                    else:
                        wrong_boxes_pseudo[pred_labels[p_id]].append(pred_scores[p_id])

    if save_preds:
        with open(frame_pred_result_save_path, 'wb') as f:
            pickle.dump(new_frame_pred_res, f)  

    thresh = []
    low_thresh = []
    for c_id, cl in enumerate(class_names):
        # ---------------- right
        X = right_boxes_pseudo[c_id+1]
        if type(right_center_num) == list:
            k1 = right_center_num[c_id]
        else:
            k1 = right_center_num
        k = min(k1, len(X))
        if k > 0:
            kmeans = KMeans(n_clusters=k)
            X = np.array(X).reshape(-1, 1)
            kmeans.fit(X)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            centroids_right = [center.item() for center in sorted(centroids)]
            logger.info('-------------------------')
            logger.info(f'{cl} right pseudo')
            logger.info('centers')
            logger.info(centroids_right)
        else:
            centroids_right = [0.3, 0.9]
        # ---------------- wrong
        X = wrong_boxes_pseudo[c_id+1]
        if type(wrong_center_num) == list:
            k2 = wrong_center_num[c_id]
        else:
            k2 = wrong_center_num
        k = min(k2, len(X))
        if k > 0:
            kmeans = KMeans(n_clusters=k)
            X = np.array(X).reshape(-1, 1)
            kmeans.fit(X)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            centroids_wrong = [center.item() for center in sorted(centroids)]
            logger.info('-------------------------')
            logger.info(f'{cl} wrong pseudo')
            logger.info('centers')
            logger.info(centroids_wrong)
        else:
            centroids_wrong = [0.3, 0.9]        

        h_t = centroids_right[-1]
        # for c_id in range(len(centroids_right)-1,-1,-1):
        #     if centroids_right[c_id] <= centroids_wrong[-1]:
        #         break
        #     h_t = centroids_right[c_id]
        thresh.append(h_t)


        l_t = centroids_wrong[0]
        # for c_id in range(len(centroids_wrong)):
        #     if centroids_wrong[c_id] >= centroids_right[0]:
        #         break
        #     l_t = centroids_wrong[c_id]
        low_thresh.append(l_t)
    model.eval()
    return thresh, low_thresh


def construct_kitti_ssl_pseudo_database_from_scenes(unlabeled_datalodar, model, thresh, low_thresh, save_dir, logger, dynamic_thresh=False):
    if dynamic_thresh:
        thresh, low_thresh_ = get_kitti_ssl_pseudo_dynamic_thresh(
            unlabeled_datalodar, model, save_dir, logger, save_preds=True,
            right_center_num=cfg.ALSSL_TRAIN.get('RIGHT_CENTER_NUM', 10), wrong_center_num=cfg.ALSSL_TRAIN.get('WRONG_CENTER_NUM', 10)
            )
        logger.info('================================')
        logger.info('new dynamic thresh')
        logger.info(thresh)
        logger.info(low_thresh)


    class_names = unlabeled_datalodar.dataset.class_names
    model.eval()
    class_name = cfg.CLASS_NAMES
    dataset_name = cfg.DATA_CONFIG.DATASET.strip('Dataset')
    assert 'Kitti' in dataset_name, "not Kitti dataset "
    box_database_save_path = Path(save_dir) / ('ssl_pseudo_database_box')
    scene_database_save_path = Path(save_dir) / ('ssl_pseudo_database_scene')
    db_info_save_path = Path(save_dir) / ('ssl_dbinfos_pseudo.pkl')
    frame_info_save_path = Path(save_dir) / ('frame_dbinfos_pseudo.pkl')

    gt_db_info_save_path = Path(save_dir) / ('ssl_dbinfos_gt.pkl')
    gt_frame_info_save_path = Path(save_dir) / ('frame_dbinfos_gt.pkl')


    with open(gt_db_info_save_path, 'rb') as f:
        gt_db_info = pickle.load(f)
    with open(gt_frame_info_save_path, 'rb') as f:
        gt_frame_info = pickle.load(f)



    box_database_save_path.mkdir(parents=True, exist_ok=True)
    scene_database_save_path.mkdir(parents=True, exist_ok=True)

    val_dataloader_iter = iter(unlabeled_datalodar)
    val_loader = unlabeled_datalodar
    total_it_each_epoch = len(unlabeled_datalodar) 

    all_db_infos = {'scene':[], 'box':{}}
    frame_dict = {}
    for cur_it in tqdm(range(total_it_each_epoch)):
        try:
            unlabeled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabeled_dataloader_iter = iter(val_loader)
            unlabeled_batch = next(unlabeled_dataloader_iter)
        
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   

                sample_idx = unlabeled_batch['frame_id'].item()
                calib = unlabeled_datalodar.dataset.get_calib(sample_idx)
                points = unlabeled_datalodar.dataset.get_lidar(sample_idx)
                info_ = unlabeled_datalodar.dataset.get_cer_info(sample_idx)
                img_shape = info_['image']['image_shape']
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = unlabeled_datalodar.dataset.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

                pseudo_obj_list = []
                all_objs_list = []
                
                pred_boxes_loc = pred_dicts[batch_inx]['pred_boxes'] #(15 or ..., 7)
                
                # if pred_boxes_loc.numel() == 0:
                #     continue
                pred_labels = pred_dicts[batch_inx]['pred_labels'] #(15 or ...)
                pred_scores = pred_dicts[batch_inx]['pred_scores'] #(15 or ...)
                # point_indices_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                #     points[:, 0:3], unlabeled_batch['gt_boxes'][0, :, :7].cpu()
                # ).numpy() 
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    points[:, 0:3], pred_boxes_loc.cpu()
                ).numpy()  
                pred_num = pred_scores.shape[0]
                obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])
                points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                info_record = {'scene': []}   
                for cl in class_names: 
                    info_record[cl] = []
                    info_record['BACK_'+cl] = []
                scene_path = scene_database_save_path / 'scenes_no_object_{}.bin'.format(sample_idx)
                with open(scene_path, 'w') as f:
                    points_no_obj.tofile(f) 
                scene_info = {'sample_idx': sample_idx, 'path': scene_path}  
                info_record['scene'].append(scene_info)  
                all_db_infos['scene'].append(scene_info)               
                for bid in range(pred_num):
                    filename = '%s_%s_%d.bin' % (sample_idx, class_name[pred_labels[bid]-1], bid)
                    back_filename = '%s_back_%s_%d.bin' % (sample_idx, class_name[pred_labels[bid]-1], bid)
                    filepath = box_database_save_path / filename
                    back_filepath = box_database_save_path / back_filename

                    box_points = points[point_indices[bid] > 0]
                    if box_points.shape[0] == 0:
                        continue
                    box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()

                    obj_points = []
                    # db_path = str(filepath.relative_to(save_dir))
                    db_path = filepath
                    # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                    obj_info = {'name': class_name[pred_labels[bid]-1], 'path': db_path, 
                                'image_idx': sample_idx, 'gt_idx': bid,
                                'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                'num_points_in_gt': box_points.shape[0],
                                'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                    
                    if pred_scores[bid] >= thresh[pred_labels[bid]-1]:
                        if sample_idx in gt_frame_info:
                            chosen_boxes = []
                            info_record_gt = gt_frame_info[sample_idx]
                            for cl in class_names:
                                # cs_box_idx = gt_db_info
                                for cs_box_idx in info_record_gt[cl]:
                                    chosen_boxes.append(gt_db_info['box'][cl][cs_box_idx]['box3d_lidar'])
                            if len(chosen_boxes) > 0:
                                chosen_boxes = np.array(chosen_boxes)
                                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(pred_boxes_loc[bid][:7].cpu().numpy().reshape(-1, 7), chosen_boxes[:, 0:7])
                                if np.max(iou2) > 0.9:
                                    continue  

                        pseudo_obj_list.append(obj_info)
                        with open(filepath, 'w') as f:
                            box_points.tofile(f)

                        if class_name[pred_labels[bid]-1] in all_db_infos['box']:
                            all_db_infos['box'][class_name[pred_labels[bid]-1]].append(obj_info)
                        else:
                            all_db_infos['box'][class_name[pred_labels[bid]-1]] = [obj_info]

                        info_record[class_name[pred_labels[bid]-1]].append(obj_info)
                    elif pred_scores[bid] <= low_thresh[pred_labels[bid]-1]:
                            cl = class_name[pred_labels[bid]-1]
                            db_path = back_filepath
                            obj_info['name'] = 'BACK_' + cl
                            obj_info['path'] = db_path
                            pseudo_obj_list.append(obj_info)
                            with open(db_path, 'w') as f:
                                box_points.tofile(f)

                            if 'BACK_' + cl in all_db_infos['box']:
                                all_db_infos['box']['BACK_' + cl].append(obj_info)
                            else:
                                all_db_infos['box']['BACK_' + cl] = [obj_info]
                            info_record['BACK_' + cl].append(obj_info)
                    elif pred_scores[bid] >= thresh[pred_labels[bid]-1]:
                        all_objs_list.append(pred_boxes_loc[bid])
                frame_dict[sample_idx] = info_record
    logger.info('Pseudo Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Pseudo Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict


def iter_reconstruct_kitti_ssl_pseudo_database_from_scenes(unlabeled_datalodar, model, thresh, low_thresh, save_dir, database_iter_del_time, logger, reconstruct_scene=False, change_scene=False, dynamic_thresh=False):
    if dynamic_thresh:
        thresh, low_thresh_ = get_kitti_ssl_pseudo_dynamic_thresh_iter(
            unlabeled_datalodar, model, save_dir, logger, save_preds=True,
            right_center_num=cfg.ALSSL_TRAIN.get('RIGHT_CENTER_NUM', 10), wrong_center_num=cfg.ALSSL_TRAIN.get('WRONG_CENTER_NUM', 10)
            )
        logger.info('================================')
        logger.info('new dynamic thresh')
        logger.info(thresh)
        logger.info(low_thresh)

    class_names = unlabeled_datalodar.dataset.class_names
    model.eval()
    class_name = cfg.CLASS_NAMES
    dataset_name = cfg.DATA_CONFIG.DATASET.strip('Dataset')
    # assert 'Kitti' in dataset_name, "not Kitti dataset "
    # print(save_dir)
    box_database_save_path = Path(save_dir) / ('ssl_pseudo_database_box')
    scene_database_save_path = Path(save_dir) / ('ssl_pseudo_database_scene')
    db_info_save_path = Path(save_dir) / ('ssl_dbinfos_pseudo.pkl')
    frame_info_save_path = Path(save_dir) / ('frame_dbinfos_pseudo.pkl')

    gt_db_info_save_path = Path(save_dir) / ('ssl_dbinfos_gt.pkl')
    gt_frame_info_save_path = Path(save_dir) / ('frame_dbinfos_gt.pkl')


    with open(gt_db_info_save_path, 'rb') as f:
        gt_db_info = pickle.load(f)
    with open(gt_frame_info_save_path, 'rb') as f:
        gt_frame_info = pickle.load(f)

    with open(db_info_save_path, 'rb') as f:
        all_db_infos = pickle.load(f)  
    with open(frame_info_save_path, 'rb') as f:
        frame_dict = pickle.load(f)

    box_database_save_path.mkdir(parents=True, exist_ok=True)
    scene_database_save_path.mkdir(parents=True, exist_ok=True)

    val_dataloader_iter = iter(unlabeled_datalodar)
    val_loader = unlabeled_datalodar
    total_it_each_epoch = len(unlabeled_datalodar) 

    for cur_it in tqdm(range(total_it_each_epoch)):
        try:
            unlabeled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabeled_dataloader_iter = iter(val_loader)
            unlabeled_batch = next(unlabeled_dataloader_iter)
        
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                sample_idx = unlabeled_batch['frame_id'].item()
                calib = unlabeled_datalodar.dataset.get_calib(sample_idx)
                points = unlabeled_datalodar.dataset.get_lidar(sample_idx)
                info_ = unlabeled_datalodar.dataset.get_cer_info(sample_idx)
                img_shape = info_['image']['image_shape']
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = unlabeled_datalodar.dataset.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

                pseudo_obj_list = []
                all_objs_list = []
                
                pred_boxes_loc = pred_dicts[batch_inx]['pred_boxes'] #(15 or ..., 7)
                
                # if pred_boxes_loc.numel() == 0:
                #     continue
                pred_labels = pred_dicts[batch_inx]['pred_labels'] #(15 or ...)
                pred_scores = pred_dicts[batch_inx]['pred_scores'] #(15 or ...)

                info_record = frame_dict[sample_idx]

                ##### to do -> put back

                if change_scene:
                    scene_path = info_record['scene'][0]['path']
                    points = np.fromfile(str(scene_path), dtype=np.float32).reshape(
                            [-1, 4])
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        points[:, 0:3], pred_boxes_loc.cpu()
                    ).numpy()  
                    pred_num = pred_scores.shape[0]
                    obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])                    
                    points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                    scene_path = scene_database_save_path / 'scenes_no_object_{}.bin'.format(sample_idx)
                    with open(scene_path, 'w') as f:
                        points_no_obj.tofile(f)                     
                elif reconstruct_scene:
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        points[:, 0:3], pred_boxes_loc.cpu()
                    ).numpy()  
                    pred_num = pred_scores.shape[0]
                    obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])
                    points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                    scene_path = scene_database_save_path / 'scenes_no_object_{}.bin'.format(sample_idx)
                    with open(scene_path, 'w') as f:
                        points_no_obj.tofile(f) 
                else:
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        points[:, 0:3], pred_boxes_loc.cpu()
                    ).numpy()  
                    pred_num = pred_scores.shape[0]

                all_objs_memory = []
                memory_boxes_loc = []
                memory_labels = []
                memory_scores = []
                memory_points = []
                for cl in class_names:
                    for obj_info in info_record[cl]:
                        all_objs_memory.append(obj_info)
                        memory_boxes_loc.append(obj_info['box3d_lidar'])
                        memory_labels.append(obj_info['name'])
                        memory_scores.append(obj_info['score'])
                        file_path = obj_info['path']
                        obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                            [-1, 4])
                        memory_points.append(obj_points)
                    info_record[cl] = []
                    info_record['BACK_' + cl] = []
                if len(memory_boxes_loc) > 0:
                    memory_boxes_loc = np.stack(memory_boxes_loc)
                    has_memory = True
                else:
                    has_memory = False
                
                # memory_scores = np.array(memory_scores)
                # has_memory = False
                if has_memory and pred_num > 0:
                    iou = iou3d_nms_utils.boxes_bev_iou_cpu(pred_boxes_loc[:, 0:7].cpu().numpy(), memory_boxes_loc[:, 0:7])
                    now_box_flag = np.array([False * pred_boxes_loc.shape[0]])
                    memory_box_flag = np.array([False * memory_boxes_loc.shape[0]])

                    now_to_memory_iou_mx = iou.max(axis=1)
                    now_to_memory_iou_mx_idx = iou.argmax(axis=1)
                    memory_to_now_iou_mx = iou.max(axis=0)
                    memory_to_now_iou_mx_idx = iou.argmax(axis=0)

                    now_to_memory_match = now_to_memory_iou_mx > 0.5
                    memory_to_now_match = memory_to_now_iou_mx > 0.5
                    
                    for m_idx, mem_obj_info in enumerate(all_objs_memory):
                        filename = '%s_%s_%d.bin' % (sample_idx, memory_labels[m_idx], len(info_record[memory_labels[m_idx]]))
                        filepath = box_database_save_path / filename
                        db_path = filepath

                        if memory_to_now_match[m_idx]:
                            now_idx = memory_to_now_iou_mx_idx[m_idx]
                            box_points = points[point_indices[now_idx] > 0]

                            if pred_scores[now_idx].cpu().numpy() < memory_scores[m_idx] or box_points.shape[0] == 0:
                                mem_obj_info['iter_time'] = 0
                                mem_points = memory_points[m_idx]
                                with open(db_path, 'w') as f:
                                    mem_points.tofile(f)                            
                                mem_obj_info['path'] = db_path
                                mem_obj_info['num_points_in_gt'] = mem_points.shape[0]
                                info_record[memory_labels[m_idx]].append(mem_obj_info)
                            else:
                                filename = '%s_%s_%d.bin' % (sample_idx, class_name[pred_labels[now_idx]-1], len(info_record[class_name[pred_labels[now_idx]-1]]))
                                filepath = box_database_save_path / filename
                                db_path = filepath
                                box_points[:, :3] -= pred_boxes_loc[now_idx, :3].cpu().numpy()
                                obj_info = {'name': class_name[pred_labels[now_idx]-1], 'path': db_path, 
                                            'image_idx': sample_idx, 'gt_idx': now_idx,
                                            'box3d_lidar': pred_boxes_loc[now_idx].cpu().numpy(), 
                                            'num_points_in_gt': box_points.shape[0],
                                            'difficulty': 0, 'bbox': None, 'score': pred_scores[now_idx].cpu().numpy(),
                                            'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                                with open(db_path, 'w') as f:
                                    box_points.tofile(f)
                                info_record[class_name[pred_labels[now_idx]-1]].append(obj_info)
                        else:
                            mem_obj_info['iter_time'] = mem_obj_info['iter_time'] + 1
                            if mem_obj_info['iter_time'] < database_iter_del_time:
                                mem_points = memory_points[m_idx]
                                with open(db_path, 'w') as f:
                                    mem_points.tofile(f)                            
                                mem_obj_info['path'] = db_path
                                mem_obj_info['num_points_in_gt'] = mem_points.shape[0]
                                info_record[memory_labels[m_idx]].append(mem_obj_info)

                for bid in range(pred_num):
                    if (has_memory and now_to_memory_match[bid]) or \
                          (pred_scores[bid].cpu().numpy() < thresh[pred_labels[bid]-1] and pred_scores[bid].cpu().numpy() > low_thresh[pred_labels[bid]-1]):
                        continue
                    filename = '%s_%s_%d.bin' % (sample_idx, class_name[pred_labels[bid]-1], len(info_record[class_name[pred_labels[bid]-1]]))
                    back_filename = '%s_back_%s_%d.bin' % (sample_idx, class_name[pred_labels[bid]-1], len(info_record['BACK_' + class_name[pred_labels[bid]-1]]))
                    filepath = box_database_save_path / filename
                    back_filepath = box_database_save_path / back_filename

                    box_points = points[point_indices[bid] > 0]

                    if box_points.shape[0] == 0:
                        continue
                    box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()

                    obj_points = []
                    # db_path = str(filepath.relative_to(save_dir))
                    # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                    obj_info = {'name': class_name[pred_labels[bid]-1], 'path': filepath, 
                                'image_idx': sample_idx, 'gt_idx': bid,
                                'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                'num_points_in_gt': box_points.shape[0],
                                'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                'flag': 'Pseudo', 'round': 0, 'iter_time':0} 

                    if pred_scores[bid] <= low_thresh[pred_labels[bid]-1]:
                        cl = class_name[pred_labels[bid]-1]
                        obj_info['name'] = 'BACK_' + cl
                        obj_info['path'] = back_filepath
                        pseudo_obj_list.append(obj_info)
                        with open(back_filepath, 'w') as f:
                            box_points.tofile(f)
                        info_record['BACK_' + cl].append(obj_info)
                    else:
                        with open(filepath, 'w') as f:
                            box_points.tofile(f)
                        info_record[class_name[pred_labels[bid]-1]].append(obj_info)
                frame_dict[sample_idx] = info_record

    for cl in class_names:
        all_db_infos['box'][cl] = []
        all_db_infos['box']['BACK_' + cl] = []
    for sample_idx in frame_dict.keys():
        info_record = frame_dict[sample_idx]
        for cl in class_names:
            all_db_infos['box'][cl].extend(info_record[cl])
            all_db_infos['box']['BACK_' + cl].extend(info_record['BACK_' + cl])
        
    logger.info('Pseudo Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Pseudo Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict

def construct_kitti_ssl_database(model, args, cfg, labeled_dataloader, unlabeled_dataloader, save_dir, logger):
    construct_kitti_ssl_gt_database_from_scenes(labeled_dataloader, save_dir, logger)
    construct_kitti_ssl_pseudo_database_from_scenes(unlabeled_dataloader, model, cfg.SSL_Thresh, logger)

