import os
import pickle
import torch

class Strategy:
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, model_list=None, logger=None):
        self.cfg = cfg
        self.active_label_dir = active_label_dir
        self.rank = rank
        self.model = model
        self.labelled_loader = labelled_loader
        self.unlabelled_loader = unlabelled_loader
        self.labelled_set = labelled_loader.dataset
        self.unlabelled_set = unlabelled_loader.dataset
        self.bbox_records = {}
        self.point_measures = ['mean', 'median', 'variance']
        self.logger = logger
        for met in self.point_measures:
            setattr(self, '{}_point_records'.format(met), {})

        self.model_list = model_list
        self.class_names = cfg.CLASS_NAMES

        if 'Kitti' in cfg.DATA_CONFIG.DATASET:
            self.pairs = list(zip(self.unlabelled_set.sample_id_list, self.unlabelled_set.kitti_infos))
        else:
            self.pairs = list(zip(self.unlabelled_set.frame_ids, self.unlabelled_set.infos))

    
    def save_active_boxes(self, selected_boxes=None, grad_embeddings=None, al_round=None, selected_sparse_frames=None):
        if selected_boxes is not None:
            if not os.path.exists(self.active_label_dir):
                os.makedirs(self.active_label_dir)
            save_path = os.path.join(self.active_label_dir, 'selected_boxes_epoch_{}_rank_{}.pkl'.format(al_round, self.rank))
            with open(save_path, 'wb') as f:
                pickle.dump({'selected_boxes': selected_boxes, 'selected_sparse_frames': selected_sparse_frames}, f)

            print('successfully saved selected boxes for epoch {} for rank {}'.format(al_round, self.rank))

        return save_path

    def load_save_boxes(self, save_path=None, grad_embeddings=None, al_round=None):
        if save_path is None:
            save_path = os.path.join(self.active_label_dir, 'selected_boxes_epoch_{}_rank_{}.pkl'.format(al_round, self.rank))
        
        with open(save_path, 'rb') as f:
            selected_boxes = pickle.load(f)['selected_boxes']
        print('successfully load selected boxes for epoch {} for rank {}'.format(al_round, self.rank))

        return selected_boxes

    def save_active_labels(self, selected_frames=None, grad_embeddings=None, al_round=None):
      
        if selected_frames is not None:
            if not os.path.exists(self.active_label_dir):
                os.makedirs(self.active_label_dir)
            save_path = os.path.join(self.active_label_dir, 'selected_frames_epoch_{}_rank_{}.pkl'.format(al_round, self.rank))
            with open(save_path, 'wb') as f:
                pickle.dump({'frame_id': selected_frames}, f)

            print('successfully saved selected frames for epoch {} for rank {}'.format(al_round, self.rank))

        return save_path

    def query(self, leave_pbar=True, al_round=None):
        pass