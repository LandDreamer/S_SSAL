import numpy as np
import pickle 
exp_name = '200pretrain_sslnaive_select'
# exp_name = '200pretrain_hssda'
# exp_name = '200pretrain_sslnaivepretrain'
# ckpt_path = "../output/kitti_selected_labels/pv_rcnn_pretrain_selected/box200/eval/epoch_80/val/default/features_res.pkl"
ckpt_path = "../output/kitti_alssl_models/pv_rcnn_pretrain_ssl_naive/box200/eval_train/epoch_no_number/val/default/features_res.pkl"
# ckpt_path = "../output/kitti_alssl_models/pv_rcnn_pretrain_pseudoscene_ssl/box200/eval_train/epoch_no_number/val/default/features_res.pkl"
# ckpt_path = "../output/kitti_alssl_models/hssda/box200/eval_train/epoch_no_number/val/default/features_res.pkl"
# ckpt_path = "../output/al_kitti/pv_rcnn_al_entropy_sslnaivepretrain_box200/box200sslnaivepretrain/eval_train/epoch_no_number/val/default/features_res.pkl"

# train_path_list = []
# selected_path_list = ['../output/al_kitti/pv_rcnn_al_entropy_sslscenepretrain_box200/box200sslscenepretrain/al_1/selected_frames_epoch_1_rank_0.pkl']

train_path_list = []
# selected_path_list = ['../output/kitti_select_samples/select_box-200/random_sample_frames.pkl']
selected_path_list = ['../output/al_kitti/pv_rcnn_al_entropy_sslnaivepretrain_box200/box200sslnaivepretrain/al_1/selected_frames_epoch_1_rank_0.pkl']
train_sample_id_list = []
selected_sample_id_list = []
for save_path in train_path_list:
    with open(save_path, 'rb') as f:
        load_list = pickle.load(f) 
        if 'frame_id' in load_list:
            load_list  = load_list['frame_id']
        train_sample_id_list.extend(load_list)

for save_path in selected_path_list:
    with open(save_path, 'rb') as f:
        load_list = pickle.load(f) 
        if 'frame_id' in load_list:
            load_list  = load_list['frame_id']
        selected_sample_id_list.extend(load_list)

with open(ckpt_path, "rb") as f:
    feat_list = pickle.load(f)
# import pdb; pdb.set_trace()
print('feat list : %d' % len(feat_list))

roi_list = []
class_list = []
score_list = []

for key, items in feat_list.items():
    roi_list.append(items["box_features"].cpu().numpy())
    pred_labels = (items["pred_labels"].cpu().numpy() - 1) * 3
    if key in train_sample_id_list:
        pred_labels += 1
    elif key in selected_sample_id_list:
        pred_labels += 2
    class_list.append(pred_labels)
    score_list.append(items["pred_scores"].cpu().numpy())

roi_feats = np.concatenate(roi_list, axis=0)
labels = np.concatenate(class_list, axis=0)
scores = np.concatenate(score_list, axis=0)
# import pdb; pdb.set_trace()
thres = 0.1

thre_ind = scores > thres

roi_feats = roi_feats[thre_ind]
labels = labels[thre_ind]


#print(labels.dtype)
#print(np.unique(labels))

assert roi_feats.shape[0] == labels.shape[0]

size = 100000

roi_list = []
class_list = []

random_seed = 2023
np.random.seed(random_seed)

for i in np.unique(labels):
    cls_ind = labels == i

    cls_ind = cls_ind.nonzero()
    # print(len(cls_ind[0]))
    if len(cls_ind[0]) > size:
        cls_ind = np.random.choice(cls_ind[0], size, replace=False)
    else:
        cls_ind = cls_ind[0]
    roi_feats_i = roi_feats[cls_ind]
    label_i = labels[cls_ind]
    #print(roi_feats_i0.shape)

    #assert False
    roi_list.append(roi_feats_i)
    class_list.append(label_i)

roi_feats = np.concatenate(roi_list, axis=0)
labels = np.concatenate(class_list, axis=0)

'''
    对roi_feats进行采样
'''


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=300)

# 将数据降维到2维
tsne_results = tsne.fit_transform(roi_feats)


# # 初始化DBSCAN模型
# dbscan = DBSCAN(eps=2.2, min_samples=30)  # 调整eps和min_samples参数以适应您的数据

# print("RoI", roi_feats.shape)

# # 进行聚类
# #labels_db = dbscan.fit_predict(roi_feats)
# labels_db = dbscan.fit_predict(tsne_results)

# print(len(labels_db))
id2name = ['Car', 'Car_select', 'Ped', 'Ped_select', 'Cyc', 'Cyc_select']

"""
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
kmeans = KMeans(n_clusters=len(id2name), random_state=42)

# 进行聚类
labels_db = kmeans.fit_predict(roi_feats)

# dbscan = DBSCAN(eps=0.5, min_samples=10) # 调整eps和min_samples参数以适应您的数据
# labels_db = dbscan.fit_predict(roi_feats)

print(len(labels_db))




class_names = np.unique(labels_db)
class_colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))






print(len(class_names))

fig, ax = plt.subplots(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    indices = np.where(labels_db == class_name)[0]
    #name = id2name[i]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'{class_name}',
                color=class_colors[i])


plt.legend()
# 添加标签和标题
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')

# 显示可视化结果
plt.savefig('./DBSCAN_visualization.png', dpi=300)  # 替换为你想要保存的文件路径
plt.show()
"""

class_names = np.unique(labels)
class_colors = []
class_colors.extend(plt.cm.rainbow(np.linspace(0, 0.2, 2)))    
class_colors.extend(plt.cm.rainbow(np.linspace(0.4, 0.6, 2)))    
class_colors.extend(plt.cm.rainbow(np.linspace(0.8, 1, 2)))    


fig, ax = plt.subplots(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    indices = np.where(labels == class_name)[0]
    name = id2name[i]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'{name}',
                color=class_colors[i])

plt.legend()
# 添加标签和标题
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')

# 显示可视化结果
plt.savefig('../vis_res/{}_pred_vis_tsne.png'.format(exp_name), dpi=300)  # 替换为你想要保存的文件路径
plt.show()