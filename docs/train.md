# Training Process Documentation

[TOC]



## Overview

This document details the training process of a 3D object detection model based on the KITTI/Waymo dataset. The entire workflow includes random initial frame selection, normal pretraining, model testing, CPSP pretraining, active learning sampling, and final model delivery.



## Training Steps

### 1. Random Initial Frame Selection

```bash
python select_frames.py --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml --extra_tag box200
```

**Note**: Please maintain consistency between CPSP and normal pretrain frames!



### 2. Normal Pretraining

```bash
python construct_ssl_database.py  --extra_tag box200 --extra_tag_2 database  --construct_set gt \
 --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml 
 
python train_selected_labels.py --extra_tag box200normalpretrain --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml  --batch_size 2
```

**Description**: Build a semi-supervised learning database for normal pretraining using labeled data and train the initial model.



### 3. Testing the Normally Pretrained Model



```bash
python test_training.py --extra_tag box200normalpretrain --cfg cfgs/kitti_selected_labels/pv_rcnn_pretrain_selected.yaml \
 --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth
```

**Description**: Evaluate the performance of the normally pretrained model on the training dataset.



### 4. CPSP Pretraining



```bash
python construct_ssl_database.py --extra_tag box200 --extra_tag_2 database --cfg cfgs/kitti_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth
```

**Description**: Build a semi-supervised learning database using the CPSP method.



```bash
python train_ssl.py --batch_size 2 --extra_tag box200 \
 --cfg cfgs/kitti_alssl_models/pv_rcnn_pretrain_cpsp.yaml \
 --pretrained_model ../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200normalpretrain/ckpt/checkpoint_epoch_80.pth
```

**Description**: Conduct semi-supervised learning model training using the CPSP method.



### 5. Active Learning Sampling



```bash
python al_sampling.py --construct_set gt --extra_tag box200cpsppretrain \
 --cfg cfgs/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200.yaml \
 --pretrained_model ../output/kitti_alssl_models/pv_rcnn_pretrain_cpsp/box200/ckpt/checkpoint_epoch_80.pth
```

**Description**: Select valuable samples for labeling using active learning methods.



### 6. Final Model Delivery



```bash
python train_selected_labels.py --extra_tag box200cpsppretrain --cfg cfgs/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200.yaml \
 --batch_size 2
```

**Description**: Train the final model using the samples selected by active learning.



```bash
python construct_ssl_database.py  --extra_tag box200cpsppretrain --extra_tag_2 database \
 --cfg cfgs/sslal_kitti/pv_rcnn_al_cal_cpsppretrain_box200_cpsp.yaml \
 --pretrained_model ../output/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200/box200cpsppretrain/ckpt/checkpoint_epoch_80.pth
```

**Description**: Build the semi-supervised learning database for the final model.



```bash
python train_ssl.py --batch_size 2 --extra_tag box200cpsppretrain \
 --cfg cfgs/sslal_kitti/pv_rcnn_al_cal_cpsppretrain_box200_cpsp.yaml \
 --pretrained_model ../output/al_kitti/pv_rcnn_al_cal_cpsppretrain_box200/box200cpsppretrain/ckpt/checkpoint_epoch_80.pth
```

**Description**: Perform semi-supervised learning training for the final model using the CPSP method.