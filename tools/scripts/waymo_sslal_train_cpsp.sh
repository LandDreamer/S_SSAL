#!/usr/bin/env bash

# PY_ARGS=${@:1}

# randomly selecting initial frames 
python select_frames.py --cfg cfgs/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000.yaml --extra_tag box5000 

# normal pretrain
python construct_ssl_database.py  --extra_tag box5000 --extra_tag_2 database  --construct_set gt \
 --cfg cfgs/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000.yaml 

# python train_selected_labels.py --extra_tag box5000normalpretrain --cfg cfgs/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000.yaml  --batch_size 2

python -m torch.distributed.launch --master_port=25333 --nproc_per_node=4 train_selected_labels.py --extra_tag box5000normalpretrain \
 --cfg cfgs/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000.yaml \
 --batch_size 8 --launcher pytorch 

# test normalpretrain model on train datasets
python test_training.py --extra_tag box5000normalpretrain --cfg cfgs/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000.yaml \
 --pretrained_model ../output/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000/box5000normalpretrain/ckpt/checkpoint_epoch_30.pth 

# cpsp pretrain
python construct_ssl_database.py  --extra_tag box5000 --extra_tag_2 database  --construct_set all \
 --cfg cfgs/waymo_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --pretrained_model ../output/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000/box5000normalpretrain/ckpt/checkpoint_epoch_30.pth

python train_ssl.py --extra_tag box5000cpsppretrain --cfg cfgs/waymo_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --batch_size 2 --pretrained_model ../output/waymo_selected_labels/pv_rcnn_pretrain_selected_box5000/box5000normalpretrain/ckpt/checkpoint_epoch_30.pth

# AL sampling
python al_sampling.py --construct_set gt --extra_tag box5000cpsppretrain \
 --cfg cfgs/al_waymo/pv_rcnn_al_cal_cpsppretrain_box5000.yaml \
 --pretrained_model ../output/waymo_alssl_models/pv_rcnn_pretrain_cpsp/box5000cpsppretrain/ckpt/checkpoint_epoch_30.pth

# Final model deliver
python -m torch.distributed.launch --master_port=25333 --nproc_per_node=4 train_selected_labels.py --extra_tag box5000cpsppretrain \
 --cfg cfgs/al_waymo/pv_rcnn_al_cal_cpsppretrain_box5000.yaml \
 --batch_size 8 --launcher pytorch 

python construct_ssl_database.py  --extra_tag box5000cpsppretrain --extra_tag_2 database \
 --cfg cfgs/sslal_waymo/pv_rcnn_al_cal_cpsppretrain_box5000_cpsp.yaml \
 --pretrained_model ../output/al_waymo/pv_rcnn_al_cal_cpsppretrain_box5000/box5000cpsppretrain/ckpt/checkpoint_epoch_30.pth

python train_ssl.py --batch_size 2 --extra_tag box5000cpsppretrain \
 --cfg cfgs/sslal_waymo/pv_rcnn_al_cal_cpsppretrain_box5000_cpsp.yaml \
 --pretrained_model ../output/al_waymo/pv_rcnn_al_cal_cpsppretrain_box5000/box5000cpsppretrain/ckpt/checkpoint_epoch_30.pth
