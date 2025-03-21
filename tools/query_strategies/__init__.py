from __future__ import absolute_import

from .random_sampling import RandomSampling_Scene
from .cal_sampling import CAL_Sampling_Scene


__factory = {
    'random': RandomSampling_Scene,
    'cal': CAL_Sampling_Scene,
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, model_list=None, logger=None):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    if model_list is not None:
        return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, model_list=model_list, logger=logger)
    else:
        return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, logger=logger)