from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import torch
from collections import OrderedDict
import pandas as pd 
import numpy as np 
from constants import get_unseen_class_ids, get_seen_class_ids
import os
from mmdet.datasets import build_dataset
from mmdet.apis.runner import copy_synthesised_weights 
from mmcv import Config


# ./tools/dist_test.sh configs/ilsvrc/faster_rcnn_r101_fpn_1x.py work_dirs/ILSVRC/epoch_12.pth 6
#  --out ilsrvc_results.pkl 
#  --syn_weights /raid/mun/codes/zero_shot_detection/zsd_ilsvrc/checkpoints/imagenet_0.6_1_0_1_w2v/classifier_best_18.pth --dataset imagenet

config_file = 'configs/ilsvrc/faster_rcnn_r101_fpn_1x.py'
syn_weights = '/raid/mun/codes/zero_shot_detection/zsd_ilsvrc/checkpoints/imagenet_0.6_1_0_1_w2v/classifier_best_18.pth'
checkpoint_file = 'work_dirs/ILSVRC/epoch_12.pth'
score_thr = 0.4
dataset_name = 'imagenet'
try:
    # os.makedirs('det_results/ILSVRC')
    # os.makedirs('det_results/coco')
    os.makedirs(f'det_results/{dataset_name}_{score_thr}')
    # os.makedirs(f'det_results/voc_{score_thr}')
except OSError:
    pass

# import pdb; pdb.set_trace()

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.test, {'test_mode': True})
model.CLASSES = dataset.CLASSES
# copy_syn_weights(syn_weights, model)
copy_synthesised_weights(model, syn_weights, dataset_name, split='177_23')
root = '/raid/mun/codes/data/ILSVRC'
# df = pd.read_csv('../MSCOCO/validation_coco_unseen_all.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name}" for file_name in file_names]
# files_path = np.array(files_path)
# img_infos
# for idx, img in enumerate(files_path[:1000]):
# import pdb; pdb.set_trace()
import random
# color = "%06x" % random.randint(0, 0xFFFFFF)
# from splits import COCO_ALL_CLASSES
# color_map = {label: (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)) for label in COCO_ALL_CLASSES}
# det_results = mmcv.load('gen_coco_results.pkl')
det_results = mmcv.load('ilsrvc_results.pkl')
# gen_filenames = [
# 'COCO_val2014_000000008676.jpg',
# 'COCO_val2014_000000012827.jpg',  'COCO_val2014_000000056430.jpg',  'COCO_val2014_000000403817.jpg',
# 'COCO_val2014_000000483108.jpg',
# 'COCO_val2014_000000012085.jpg',  'COCO_val2014_000000027371.jpg',  'COCO_val2014_000000069411.jpg',
# 'COCO_val2014_000000428454.jpg',
# 'COCO_val2014_000000553721.jpg']

# zsd = [
#     'COCO_val2014_000000052066.jpg',  'COCO_val2014_000000058225.jpg' ,
#     'COCO_val2014_000000128644.jpg' , 'COCO_val2014_000000350073.jpg' ,'COCO_val2014_000000519299.jpg',
#     'COCO_val2014_000000054277.jpg',  'COCO_val2014_000000101088.jpg' , 'COCO_val2014_000000171058.jpg',  
#     'COCO_val2014_000000512455.jpg'  ,'COCO_val2014_000000572517.jpg',
# ]
# start = 3000
inds = np.random.permutation(np.arange(len(dataset.img_infos)))[:1000]
img_infos = dataset.img_infos
# [start:start+1000]
# [:1000]
for idx in inds:
    img = f"{root}/{img_infos[idx]['filename']}"

    # data_split = 'train' if 'train' in img_id else 'val'

    # filename = f'Data/DET/{data_split}/{img_id}.JPEG'
    # if info['filename'] in zsd:
    # result = inference_detector(model, img)
    result = det_results[idx]#inference_detector(model, img)
    # import pdb; pdb.set_trace()
    out_file = f"det_results/{dataset_name}_{score_thr}/{img.split('/')[-1]}"
    show_result(f"{img}", result, model.CLASSES, out_file=out_file,show=False, score_thr=score_thr, dataset=dataset_name)
    print(f"[{idx:03}/{len(img_infos)}]")
    
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# copy_syn_weights(syn_weights, model)
# copy_synthesised_weights(model, syn_weights)

# root = '/raid/mun/codes/data/pascalv_voc/VOCdevkit/'
# df = pd.read_csv('../VOC/testval_voc07_unseen.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name[14:]}" for file_name in file_names]
# files_path = np.array(files_path)
# for idx, img in enumerate(files_path[:100]):

#     result = inference_detector(model, img)
#     out_file = f"det_results/voc/{img.split('/')[-1]}"
#     show_result(img, result, model.CLASSES, out_file=out_file,show=False, score_thr=0.3)
#     print(f"[{idx:03}/{len(files_path)}]")

# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --syn_weights /raid/mun/codes/zero_shot_det
# ection/cvpr18xian_pascal_voc/checkpoints/VOC/classifier_best.pth --out voc_detections.p