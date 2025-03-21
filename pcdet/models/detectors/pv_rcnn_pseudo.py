import os
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict
from collections import defaultdict, OrderedDict

from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN


class PVRCNN_Pseudo(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        model_cfg_ema = copy.deepcopy(model_cfg)
        dataset_ema = copy.deepcopy(dataset)
        model_cfg_stu = copy.deepcopy(model_cfg)
        dataset_ema_stu = copy.deepcopy(dataset)
        
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        for param in self.parameters():
            param.detach_()
            param.requires_grad = False
        self.module_list = self.build_networks()

        self.pv_rcnn = PVRCNN(model_cfg=model_cfg_stu, num_class=num_class, dataset=dataset_ema_stu)
        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_ema, num_class=num_class, dataset=dataset_ema)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
            param.requires_grad = False

        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.my_global_step = 0
        if 'SSL_Thresh' in model_cfg:
            self.unlabeled_weight = model_cfg.SSL_Thresh.get('UNLABELED_WEIGHT', 0.8)
            self.ssl_weight = model_cfg.SSL_Thresh
        else:
            self.unlabeled_weight = 0.8

    def forward(self, batch_dict):
        if self.training:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)
            if 'mask' in batch_dict:
                mask = batch_dict['mask'].view(-1)

                labeled_mask = torch.nonzero(mask).squeeze(1).long()
                unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            else:
                labeled_mask = None
                unlabeled_mask = None

            loss, tb_dict, disp_dict = self.get_training_loss(labeled_mask, unlabeled_mask)
            ret_dict = {
                'loss': loss,
                'rcnn_reg_gt': self.pv_rcnn.roi_head.forward_ret_dict['rcnn_reg_gt'],
                'rcnn_cls_gt': self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'],
                'rcnn_cls': batch_dict['rcnn_cls'],
                'rcnn_reg': batch_dict['rcnn_reg'],
                'rpn_preds': batch_dict['rpn_preds']
            }
            return ret_dict, tb_dict, disp_dict
        else:
            for cur_module in self.pv_rcnn_ema.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, labeled_mask, unlabeled_mask):
        disp_dict = {}
        # import pdb; pdb.set_trace()
        loss_rpn, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
        loss_rcnn, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)
        batch_size = loss_rpn.shape[0]
        # import pdb; pdb.set_trace()
        if labeled_mask is not None and unlabeled_mask is not None:
            loss_rpn = loss_rpn[labeled_mask, ...].sum() + loss_rpn[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss_point = loss_point[labeled_mask, ...].sum() + loss_point[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss_rcnn = loss_rcnn[labeled_mask, ...].sum() + loss_rcnn[unlabeled_mask, ...].sum() * self.unlabeled_weight
        else:
            loss_rpn = loss_rpn.sum()
            loss_point = loss_point.sum()
            loss_rcnn = loss_rcnn.sum()
        loss = loss_rpn + loss_point + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.pv_rcnn.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
        loss = loss / batch_size
        return loss, tb_dict, disp_dict

    @torch.no_grad()
    def update_global_step(self):
        self.my_global_step += 1
        ema_keep_rate = 0.9996
        change_global_step = 2000
        if self.my_global_step < change_global_step:
            keep_rate = (ema_keep_rate - 0.5) / change_global_step * self.my_global_step + 0.5
        else:
            keep_rate = ema_keep_rate

        student_model_dict = self.pv_rcnn.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.pv_rcnn_ema.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise NotImplementedError
        self.pv_rcnn_ema.load_state_dict(new_teacher_dict)

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)

        if 'version' in checkpoint:
            if logger is not None:
                logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
        update_model_state = {}
        for key, val in model_state_disk.items():
            # if 'pfe' in key and '50' in filename:
            #     import pdb; pdb.set_trace()
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
        for key, val in model_state_disk.items():
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val            

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)
        for key in state_dict:
            if key not in update_model_state:
                if logger is not None:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        if logger is not None:
            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
