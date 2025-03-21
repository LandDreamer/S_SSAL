
import numpy as np
import torch
import pickle
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_selected_dataloader, build_active_dataloader
from pcdet.models import build_network
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from pcdet.models import load_data_to_gpu
from pcdet.utils import box_utils, common_utils, calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from sklearn.cluster import KMeans
from ssl_utils.kitti_ssl_tools import get_kitti_ssl_pseudo_dynamic_thresh, get_kitti_ssl_pseudo_dynamic_thresh_iter


def construct_waymo_ssl_gt_database_from_scenes(labeled_dataloader, save_dir, logger):
    class_names = labeled_dataloader.dataset.class_names
    dataset_name = cfg.DATA_CONFIG.DATASET.strip('Dataset')
    assert 'Waymo' in dataset_name, "not Waymo dataset "
    box_database_save_path = Path(save_dir) / ('ssl_gt_database_box')
    scene_database_save_path = Path(save_dir) / ('ssl_gt_database_scene')
    db_info_save_path = Path(save_dir) / ('ssl_dbinfos_gt.pkl')
    frame_info_save_path = Path(save_dir) / ('frame_dbinfos_gt.pkl')

    box_database_save_path.mkdir(parents=True, exist_ok=True)
    scene_database_save_path.mkdir(parents=True, exist_ok=True) 

    all_db_infos = {'scene':[], 'box':{}}

    labeled_dataset = labeled_dataloader.dataset
    labeled_infos = labeled_dataset.infos
    frame_dict = {}
    point_offset_cnt = 0
    stacked_gt_points = []
    for k in tqdm(range(len(labeled_infos))):
        info = labeled_infos[k]
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = labeled_dataset.get_lidar(sequence_name, sample_idx)
        frame_id = info['frame_id']
        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        gt_boxes = annos['gt_boxes_lidar']

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  
        
        points_no_obj = box_utils.remove_points_in_boxes3d(points, gt_boxes)
        scene_path = scene_database_save_path / 'scenes_no_object_{}_{}_{}.bin'.format(k, sequence_name, sample_idx)
        with open(scene_path, 'w') as f:
            points_no_obj.tofile(f) 
        scene_info = {'sample_idx': frame_id, 'path': scene_path}  
        info_record = {'scene': [len(all_db_infos['scene'])]}
        for cl in class_names: 
            info_record[cl] = []   
            info_record['BACK_'+cl] = []  
        all_db_infos['scene'].append(scene_info)
        for i in range(num_obj):
            if names[i] not in class_names:
                continue
            filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
            filepath = box_database_save_path / filename
            gt_points = points[point_indices[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
            
            db_path = filepath
            db_info = {'name': names[i], 'path': db_path, 'image_idx': frame_id, 'gt_idx': i, 'sequence_name': sequence_name,
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i], 
                        'flag': 'GT', 'round': 0}

            stacked_gt_points.append(gt_points)
            db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
            point_offset_cnt += gt_points.shape[0] 

            if names[i] in all_db_infos['box']:
                all_db_infos['box'][names[i]].append(db_info)
            else:
                all_db_infos['box'][names[i]] = [db_info]

            
            info_record[names[i]].append(db_info)
        frame_dict[frame_id] = info_record
    logger.info('Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict

def construct_waymo_ssl_back_database_from_labeled_scenes(labeled_dataloader, model, thresh, save_dir, logger):
    class_names = labeled_dataloader.dataset.class_names
    model.eval()
    class_name = cfg.CLASS_NAMES
    dataset_name = cfg.DATA_CONFIG.DATASET.strip('Dataset')
    assert 'Waymo' in dataset_name, "not Waymo dataset "
    box_database_save_path = Path(save_dir) / ('ssl_gt_database_box')
    gt_db_info_save_path = Path(save_dir) / ('ssl_dbinfos_gt.pkl')
    gt_frame_info_save_path = Path(save_dir) / ('frame_dbinfos_gt.pkl')

    with open(gt_db_info_save_path, 'rb') as f:
        gt_db_info = pickle.load(f)
    with open(gt_frame_info_save_path, 'rb') as f:
        gt_frame_info = pickle.load(f)

    train_dataloader_iter = iter(labeled_dataloader)
    train_loader = labeled_dataloader
    total_it_each_epoch = len(labeled_dataloader) 

    all_db_infos = gt_db_info
    frame_dict = gt_frame_info
    for cl in class_names:
        all_db_infos['box']['BACK_'+cl] = []
    for cur_it in tqdm(range(total_it_each_epoch)):
        # if cur_it > 5:
        #     break
        try:
            labeled_batch = next(train_dataloader_iter)
        except StopIteration:
            labeled_dataloader_iter = iter(train_loader)
            labeled_batch = next(labeled_dataloader_iter)
        with torch.no_grad():
            load_data_to_gpu(labeled_batch)
            # import pdb; pdb.set_trace()
            pred_dicts, _ = model(labeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                frame_id = labeled_batch['frame_id'].item()
                if frame_id not in frame_dict:
                    frame_box_idx = {cl:-1 for cl in class_names}
                else:
                    frame_box_idx = frame_dict[frame_id]

                sample_idx = labeled_batch['sample_idx'].item()
                frame_id_info = frame_id.split('_')
                # seq_idx = '_'.join(frame_id_info[:-1])
                sequence_name = labeled_batch['sequence_name'][0][0]

                points = labeled_dataloader.dataset.get_lidar(sequence_name, sample_idx)

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

                back_flag = pred_scores < thresh
                pred_scores = pred_scores[back_flag]
                pred_boxes_loc = pred_boxes_loc[back_flag]
                pred_labels = pred_labels[back_flag]


                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    points[:, 0:3], pred_boxes_loc.cpu()
                ).numpy()  


                # import pdb; pdb.set_trace()
                pred_num = pred_scores.shape[0]
                back_num = {}
                for cl in class_names:
                    back_num[cl] = 0
                for bid in range(pred_num):
                    cl = class_name[pred_labels[bid]-1]
                    filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, 'BACK_'+cl, back_num[cl])
                    filepath = box_database_save_path / filename

                    box_points = points[point_indices[bid] > 0]
                    if box_points.shape[0] == 0:
                        continue
                    back_num[cl] += 1
                    box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()
                    with open(filepath, 'w') as f:
                        box_points.tofile(f)
                    
                    db_path = filepath
                    # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                    obj_info = {'name': 'BACK_'+cl, 'path': db_path, 
                                'image_idx': frame_id, 'gt_idx': bid,
                                'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                'num_points_in_gt': box_points.shape[0],
                                'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                    all_db_infos['box']['BACK_'+cl].append(obj_info)
                # import pdb; pdb.set_trace()
                if frame_id in frame_dict:
                    frame_dict[frame_id]['BACK_'+cl] = [back_num[cl]-1]
                else:
                    frame_dict[frame_id] = {
                        'scene':[], 'Vehicle': [], 'Pedestrian': [], 'Cyclist': []
                    }
                    for cl in class_names:
                        back_num['BACK_'+cl] = [back_num[cl]-1]
        # if cur_it > 100:
        #     break
    for k, v in all_db_infos['box'].items():
        logger.info('Database box %s: %d' % (k, len(v)))

    with open(gt_db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   
    with open(gt_frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict



def construct_waymo_ssl_pseudo_database_from_scenes(unlabeled_datalodar, model, thresh, low_thresh, save_dir, logger, selected_frames=None, get_boxes=True, dynamic_thresh=False):
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
    assert 'Waymo' in dataset_name, "not Waymo dataset "
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
        if selected_frames is not None and unlabeled_batch['frame_id'][0] not in selected_frames:
            continue
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                frame_id = unlabeled_batch['frame_id'].item()
                sample_idx = unlabeled_batch['sample_idx'].item()
                frame_id_info = frame_id.split('_')
                # seq_idx = '_'.join(frame_id_info[:-1])
                sequence_name = unlabeled_batch['sequence_name'][0][0]

                points = unlabeled_datalodar.dataset.get_lidar(sequence_name, sample_idx)

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

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    pred_boxes_loc[:, 0:7].unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

                pred_num = pred_scores.shape[0]
                obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])
                points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                info_record = {'scene': []}   
                for cl in class_names: 
                    info_record[cl] = []
                    info_record['BACK_'+cl] = []
                scene_path = scene_database_save_path / 'scenes_no_object_{}_{}.bin'.format(sequence_name, sample_idx)
                with open(scene_path, 'w') as f:
                    points_no_obj.tofile(f) 
                scene_info = {'sample_idx': frame_id, 'path': scene_path}  
                info_record['scene'].append(scene_info)  
                all_db_infos['scene'].append(scene_info)  
                if get_boxes:
                    for bid in range(pred_num):
                        filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, class_name[pred_labels[bid]-1], bid)
                        filepath = box_database_save_path / filename

                        box_points = points[point_indices[bid] > 0]
                        box_points_2 = points[box_idxs_of_pts == bid]
                        if box_points.shape[0] == 0:
                            continue
                        box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()

                        obj_points = []
                        # db_path = str(filepath.relative_to(save_dir))
                        db_path = filepath
                        # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                        obj_info = {'name': class_name[pred_labels[bid]-1], 'path': db_path, 
                                    'image_idx': frame_id, 'gt_idx': bid,
                                    'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                    'num_points_in_gt': box_points.shape[0],
                                    'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                    'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                        
                        if pred_scores[bid] >= thresh[pred_labels[bid]-1]:

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
                            obj_info['name'] = 'BACK_' + cl
                            pseudo_obj_list.append(obj_info)
                            with open(filepath, 'w') as f:
                                box_points.tofile(f)

                            if 'BACK_' + cl in all_db_infos['box']:
                                all_db_infos['box']['BACK_' + cl].append(obj_info)
                            else:
                                all_db_infos['box']['BACK_' + cl] = [obj_info]
                            info_record['BACK_' + cl].append(obj_info)
                        elif pred_scores[bid] >= thresh[pred_labels[bid]-1]:
                            all_objs_list.append(pred_boxes_loc[bid])
                frame_dict[frame_id] = info_record
    logger.info('Pseudo Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Pseudo Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict


def iter_reconstruct_waymo_ssl_pseudo_database_from_scenes(unlabeled_datalodar, model, thresh, low_thresh, save_dir, database_iter_del_time, logger, selected_frames=None, reconstruct_scene=False, change_scene=False, dynamic_thresh=False, keep_same_frame=True):
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
    assert 'Waymo' in dataset_name, "not Waymo dataset "
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

        if selected_frames is not None and unlabeled_batch['frame_id'][0] not in selected_frames:
            continue        
        if unlabeled_batch['frame_id'][0] not in frame_dict and keep_same_frame:
            continue
        with torch.no_grad():
            load_data_to_gpu(unlabeled_batch)
            pred_dicts, _ = model(unlabeled_batch)
            batch_size = len(pred_dicts)
            for batch_inx in range(len(pred_dicts)):   
                frame_id = unlabeled_batch['frame_id'].item()
                sample_idx = unlabeled_batch['sample_idx'].item()
                frame_id_info = frame_id.split('_')
                # seq_idx = '_'.join(frame_id_info[:-1])
                sequence_name = unlabeled_batch['sequence_name'][0][0]

                points = unlabeled_datalodar.dataset.get_lidar(sequence_name, sample_idx)

                pseudo_obj_list = []
                all_objs_list = []
                
                pred_boxes_loc = pred_dicts[batch_inx]['pred_boxes'] #(15 or ..., 7)
                
                # if pred_boxes_loc.numel() == 0:
                #     continue
                pred_labels = pred_dicts[batch_inx]['pred_labels'] #(15 or ...)
                pred_scores = pred_dicts[batch_inx]['pred_scores'] #(15 or ...)


                if frame_id in frame_dict:
                    info_record = frame_dict[frame_id]
                else:
                    info_record = {'scene': []}   
                    for cl in class_names: 
                        info_record[cl] = []
                        info_record['BACK_'+cl] = []
                ##### to do -> put back

                if change_scene:
                    scene_path = info_record['scene'][0]['path']
                    points = np.fromfile(str(scene_path), dtype=np.float32).reshape(
                            [-1, 5])
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        points[:, 0:3], pred_boxes_loc.cpu()
                    ).numpy()  
                    pred_num = pred_scores.shape[0]
                    obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])                    
                    points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                    scene_path = scene_database_save_path / 'scenes_no_object_{}_{}.bin'.format(sequence_name, sample_idx)
                    with open(scene_path, 'w') as f:
                        points_no_obj.tofile(f)                     
                elif reconstruct_scene:
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        points[:, 0:3], pred_boxes_loc.cpu()
                    ).numpy()  
                    pred_num = pred_scores.shape[0]
                    obj_mask = np.array([(pred_scores[idx] > low_thresh[pred_labels[idx]-1]).item() for idx in range(pred_labels.shape[0])])
                    points_no_obj = box_utils.remove_points_in_boxes3d(points, pred_boxes_loc[obj_mask].cpu())
                    scene_path = scene_database_save_path / 'scenes_no_object_{}_{}.bin'.format(sequence_name, sample_idx)
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
                            [-1, 5])
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
                        filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, memory_labels[m_idx], len(info_record[memory_labels[m_idx]]))
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
                                filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, class_name[pred_labels[now_idx]-1], len(info_record[class_name[pred_labels[now_idx]-1]]))
                                filepath = box_database_save_path / filename
                                db_path = filepath
                                box_points[:, :3] -= pred_boxes_loc[now_idx, :3].cpu().numpy()
                                obj_info = {'name': class_name[pred_labels[now_idx]-1], 'path': db_path, 
                                            'image_idx': frame_id, 'gt_idx': now_idx,
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
                    if pred_scores[bid].cpu().numpy() < low_thresh[pred_labels[bid]-1]:
                        cl = class_name[pred_labels[bid]-1]
                        filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, 'BACK_'+cl, len(info_record['BACK_'+cl]))
                        filepath = box_database_save_path / filename
                        db_path = filepath

                        box_points = points[point_indices[bid] > 0]

                        if box_points.shape[0] == 0:
                            continue
                        box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()

                        obj_points = []
                        # db_path = str(filepath.relative_to(save_dir))
                        db_path = filepath
                        # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                        obj_info = {'name': 'BACK_'+cl, 'path': db_path, 
                                    'image_idx': frame_id, 'gt_idx': bid,
                                    'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                    'num_points_in_gt': box_points.shape[0],
                                    'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                    'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                        
                        with open(db_path, 'w') as f:
                            box_points.tofile(f)
                        info_record['BACK_'+cl].append(obj_info)
                        continue
                    if (has_memory and now_to_memory_match[bid]) or pred_scores[bid].cpu().numpy() < thresh[pred_labels[bid]-1]:
                        continue
                    filename = '%s_%s_%s_%d.bin' % (sequence_name, sample_idx, class_name[pred_labels[bid]-1], len(info_record[class_name[pred_labels[bid]-1]]))
                    filepath = box_database_save_path / filename
                    db_path = filepath


                    box_points = points[point_indices[bid] > 0]

                    if box_points.shape[0] == 0:
                        continue
                    # import pdb; pdb.set_trace()
                    box_points[:, :3] -= pred_boxes_loc[bid, :3].cpu().numpy()

                    obj_points = []
                    # db_path = str(filepath.relative_to(save_dir))
                    db_path = filepath
                    # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'bbox', 'score'])
                    obj_info = {'name': class_name[pred_labels[bid]-1], 'path': db_path, 
                                'image_idx': frame_id, 'gt_idx': bid,
                                'box3d_lidar': pred_boxes_loc[bid].cpu().numpy(), 
                                'num_points_in_gt': box_points.shape[0],
                                'difficulty': 0, 'bbox': None, 'score': pred_scores[bid].cpu().numpy(),
                                'flag': 'Pseudo', 'round': 0, 'iter_time':0} 
                    
                    with open(db_path, 'w') as f:
                        box_points.tofile(f)
                    info_record[class_name[pred_labels[bid]-1]].append(obj_info)
                
                frame_dict[frame_id] = info_record
    for cl in class_names:
        all_db_infos['box'][cl] = []
    for frame_id in frame_dict.keys():
        info_record = frame_dict[frame_id]
        for cl in class_names:
            all_db_infos['box'][cl].extend(info_record[cl])
        
    logger.info('Pseudo Database scene: %d' % (len(all_db_infos['scene'])))
    for k, v in all_db_infos['box'].items():
        logger.info('Pseudo Database box %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   
    with open(frame_info_save_path, 'wb') as f:
        pickle.dump(frame_dict, f)
    return all_db_infos, frame_dict

