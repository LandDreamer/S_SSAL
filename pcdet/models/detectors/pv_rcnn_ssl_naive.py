import os
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict
from collections import defaultdict, OrderedDict

from .detector3d_template import Detector3DTemplate
from .pv_rcnn_ssl import PVRCNN_SSL
from pcdet.datasets.augmentor.augmentor_utils import (
    random_flip_along_x, random_flip_along_y, global_rotation, global_scaling, global_scaling_with_roi_boxes,
    random_image_flip_horizontal, random_local_translation_along_x, random_local_translation_along_y,
    random_local_translation_along_z, local_scaling, local_rotation, 
    random_flip_along_x_bbox, random_flip_along_y_bbox, global_rotation_bbox, global_scaling_bbox
)
from pcdet.ops.iou3d_nms import iou3d_nms_utils

class PVRCNN_SSL_Naive(PVRCNN_SSL):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'SSL_Thresh' in model_cfg:
            self.thresh = model_cfg.SSL_Thresh.THRESH
            self.sem_thresh = model_cfg.SSL_Thresh.SEM_THRESH
            self.unlabeled_supervise = model_cfg.SSL_Thresh.UNLABELED_SUPERVISE
            self.unlabeled_weight = model_cfg.SSL_Thresh.UNLABELED_WEIGHT
            self.no_nms = model_cfg.SSL_Thresh.NO_NMS
            self.supervise_mode = model_cfg.SSL_Thresh.SUPERVISE_MODE

    def forward(self, batch_dict):
        if self.training:
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                # import pdb; pdb.set_trace()
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema)
                # import pdb; pdb.set_trace()
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                # import pdb; pdb.set_trace()
                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        continue


                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))

                    valid_inds = pseudo_score > conf_thresh.squeeze()

                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]

                    # if len(valid_inds) > max_box_num:
                    #     _, inds = torch.sort(pseudo_score, descending=True)
                    #     inds = inds[:max_box_num]
                    #     pseudo_box = pseudo_box[inds]
                    #     pseudo_label = pseudo_label[inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                    # pseudo_scores.append(pseudo_score)
                    # pseudo_labels.append(pseudo_label)
                # import pdb; pdb.set_trace()
                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]
                # import pdb; pdb.set_trace()
                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes
                # import pdb; pdb.set_trace()
                if 'flip_x' in batch_dict:
                    batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                    )
                if 'flip_y' in batch_dict:
                    batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_y_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_y'][unlabeled_mask, ...]
                    )
                if 'noise_rot' in batch_dict:
                    batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['noise_rot'][unlabeled_mask, ...]
                    )
                if 'noise_scale' in batch_dict:
                    batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['noise_scale'][unlabeled_mask, ...]
                    )
                # import pdb; pdb.set_trace()


            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}

            loss_rpn, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            
            # import pdb; pdb.set_trace()
            if not self.unlabeled_supervise:
                loss_rpn = loss_rpn[labeled_mask, ...].sum()
            else:
                loss_rpn = loss_rpn[labeled_mask, ...].sum() + loss_rpn[unlabeled_mask, ...].sum() * self.unlabeled_weight
                loss_point = loss_point[labeled_mask, ...].sum() + loss_point[unlabeled_mask, ...].sum() * self.unlabeled_weight
                loss_rcnn = loss_rcnn[labeled_mask, ...].sum() + loss_rcnn[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss = loss_rpn + loss_point + loss_rcnn

            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn_ema.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict)

            return pred_dicts, recall_dicts
