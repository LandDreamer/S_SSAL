import random
from .strategy import Strategy
import tqdm
import torch
from pcdet.models import load_data_to_gpu


class RandomSampling_Scene(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, logger=None):
        super(RandomSampling_Scene, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, logger=logger)

    def query(self, leave_pbar=True, al_round=None):
        if len(self.bbox_records) == 0:
            
            val_dataloader_iter = iter(self.unlabelled_loader)
            val_loader = self.unlabelled_loader
            total_it_each_epoch = len(self.unlabelled_loader)
            all_frames = []

            if self.rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                                desc='going through_unlabelled_set_epoch_%d' % al_round, dynamic_ncols=True)
            self.model.eval()
            for cur_it in range(total_it_each_epoch):
                try:
                    unlabeled_batch = next(val_dataloader_iter)
                except StopIteration:
                    unlabeled_dataloader_iter = iter(val_loader)
                    unlabeled_batch = next(unlabeled_dataloader_iter)
                with torch.no_grad():
                    load_data_to_gpu(unlabeled_batch)
                    for batch_inx in range(len(unlabeled_batch['frame_id'])):
                        all_frames.append((unlabeled_batch['frame_id'][batch_inx], unlabeled_batch['gt_names'][0][batch_inx]))

                if self.rank == 0:
                    pbar.update()
                    pbar.refresh()

            if self.rank == 0:
                pbar.close()

        random.shuffle(all_frames)
        selected_frames = all_frames#[:self.cfg.ACTIVE_TRAIN.SELECT_NUMS]
        selected_frames_by_boxes = []
        thresh = self.cfg.ALSSL_TRAIN.get('SELECT_BOX_NUMS', 0)
        if thresh > 0:
            s = 0
            for idx in range(len(selected_frames)):
                num_boxes = len(selected_frames[idx][1])
                s += num_boxes
                if s > thresh:
                    break
                selected_frames_by_boxes.append(selected_frames[idx][0])
            selected_frames = selected_frames_by_boxes
        else:
            selected_frames = [frame[0] for frame in selected_frames]         
        
        
        return selected_frames