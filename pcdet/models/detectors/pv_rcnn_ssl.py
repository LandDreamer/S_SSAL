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


class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        model_cfg_ema = copy.deepcopy(model_cfg)
        dataset_ema = copy.deepcopy(dataset)
        model_cfg_stu = copy.deepcopy(model_cfg)
        dataset_ema_stu = copy.deepcopy(dataset)
        
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.pv_rcnn = PVRCNN(model_cfg=model_cfg_stu, num_class=num_class, dataset=dataset_ema_stu)
        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_ema, num_class=num_class, dataset=dataset_ema)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()

        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.my_global_step = 0

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
        
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

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
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
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
