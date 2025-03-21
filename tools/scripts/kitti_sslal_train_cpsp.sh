#!/usr/bin/env bash

# PY_ARGS=${@:1}

# # Randomly selecting initial frames 
# python select_frames.py --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml --extra_tag box200 

# # Normal pretrain
# python construct_ssl_database.py  --extra_tag box200 --extra_tag_2 database  --construct_set gt \
#  --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml 

# python train_selected_labels.py --extra_tag box200normalpretrain --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml  --batch_size 2

# # Test normalpretrain model on train datasets
# python test_training.py --extra_tag box200normalpretrain --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml \
#  --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth 

# CPSP pretrain
python construct_ssl_database.py --extra_tag box200 --extra_tag_2 database --cfg cfgs/kitti_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth  

python train_ssl.py --batch_size 2 --extra_tag box200 \
 --cfg cfgs/kitti_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth 

# AL sampling
python al_sampling.py --construct_set gt --extra_tag box200cpsppretrain \
 --cfg cfgs/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200.yaml \
 --pretrained_model ../output/kitti_alssl_models/pv_rcnn_pretrain_cpsp/box200/ckpt/checkpoint_epoch_80.pth

# Final model deliver
python train_selected_labels.py --extra_tag box200cpsppretrain --cfg cfgs/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200.yaml \
 --batch_size 2  

python construct_ssl_database.py  --extra_tag box200cpsppretrain --extra_tag_2 database \
 --cfg cfgs/sslal_kitti/pv_rcnn_al_cal_cpsppretrain_box200_cpsp.yaml \
 --pretrained_model ../output/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200/box200cpsppretrain/ckpt/checkpoint_epoch_80.pth

python train_ssl.py --batch_size 2 --extra_tag box200cpsppretrain \
 --cfg cfgs/sslal_kitti/pv_rcnn_al_cal_cpsppretrain_box200_cpsp.yaml \
 --pretrained_model ../output/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200/box200cpsppretrain/ckpt/checkpoint_epoch_80.pth


