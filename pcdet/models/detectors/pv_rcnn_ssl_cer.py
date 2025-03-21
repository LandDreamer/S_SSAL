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
from ...utils import box_coder_utils, common_utils, loss_utils

class PVRCNN_SSL_CER(PVRCNN_SSL):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'SSL_Thresh' in model_cfg:
            self.thresh = model_cfg.SSL_Thresh.THRESH
            self.sem_thresh = model_cfg.SSL_Thresh.SEM_THRESH
            self.unlabeled_supervise = model_cfg.SSL_Thresh.UNLABELED_SUPERVISE
            self.unlabeled_weight = model_cfg.SSL_Thresh.UNLABELED_WEIGHT
            self.no_nms = model_cfg.SSL_Thresh.NO_NMS
            self.supervise_mode = model_cfg.SSL_Thresh.SUPERVISE_MODE
        self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.cer_thresh = [0.99, 0.9, 0.95]

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
            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema)
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                valid_box_num = []
                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        valid_box_num.append(0)
                        continue

                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))

                    valid_inds = pseudo_score > conf_thresh.squeeze()
                    valid_box_num.append(valid_inds.sum().item())
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
                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]
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


            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}

            loss_rpn, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            
            
            if not self.unlabeled_supervise:
                loss_rpn = loss_rpn[labeled_mask, ...].sum()
            else:
                loss_rpn = loss_rpn[labeled_mask, ...].sum()
                loss_point = loss_point[labeled_mask, ...].sum()
                loss_rcnn = loss_rcnn[labeled_mask, ...].sum()
            # import pdb; pdb.set_trace()
            # pred_labels = batch_dict['roi_labels']
            pred_scores = batch_dict['batch_cls_preds'].sigmoid()
            pred_boxes = batch_dict['batch_box_preds']
            loss_uncer = self.get_unlabeled_loss(pred_boxes, pred_scores, batch_dict['gt_boxes'], pred_dicts, unlabeled_mask, valid_box_num) * self.unlabeled_weight
            loss_rcnn += loss_uncer * self.unlabeled_weight
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

    def get_unlabeled_loss(self, pred_boxes, pred_scores, ema_boxes, pred_dicts, unlabeled_mask, valid_box_num):
        loss = 0
        if len(valid_box_num) == 0:
            return 0
        for idx, ind in enumerate(unlabeled_mask):
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(ema_boxes[ind][:valid_box_num[idx], :7].cpu(), pred_boxes[ind].cpu())
            iou_mx, iou_mx_id = iou.max(1)
            consis_boxes = pred_boxes[ind][iou_mx_id]
            iou_flag = iou_mx > 0
            consis_boxes_1 = ema_boxes[ind][:valid_box_num[idx], :7][iou_flag]
            consis_boxes_2 = consis_boxes[iou_flag]
            if consis_boxes_1.shape[0] > 0 and consis_boxes_1.shape[0] > 0:
                rcnn_loss_reg = self.reg_loss_func(
                    consis_boxes_1.unsqueeze(dim=0),
                    consis_boxes_2.unsqueeze(dim=0),
                )  # [B, M, 7]
                loss += rcnn_loss_reg.sum()
            valid_boxes = pred_scores[ind] > 0.99
            valid_score = pred_scores[ind][valid_boxes]
            loss += -valid_score.log2().sum()     
        

        return loss
    