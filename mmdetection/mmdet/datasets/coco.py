import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
import pandas as pd 
import os.path
from splits import get_unseen_class_ids, get_seen_class_ids
# from mmdet.apis import get_root_logger

@DATASETS.register_module


class CocoDataset(CustomDataset):
    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    #            'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    #            'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    #            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat',
        'chair', 'cow', 'diningtable', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'tvmonitor', 'car', 'dog', 'sofa', 'train')



    def _filter_classes(self, classes_to_load=None, exclude_all=False, split='65_15'):
        """
        exclude_all: exludes all images where any object of exluded categories is present
        """

        img_ids = self.coco.getImgIds()
        cat_ids = np.array(self.coco.getCatIds())

        unseen_class_labels, seen_class_labels = get_unseen_class_ids('coco', split=split)-1, get_seen_class_ids('coco', split=split) -1
        seen_classes_cat_ids = cat_ids[seen_class_labels]
        unseen_classes_cat_ids = cat_ids[unseen_class_labels]
        self.cat_to_load = cat_ids
        if classes_to_load == 'seen':
            self.cat_to_load = seen_classes_cat_ids
        elif classes_to_load == 'unseen':
            self.cat_to_load = unseen_classes_cat_ids


        images_ids_to_exclude = []
        images_ids_to_load = []
        for index in range(len(img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=img_ids[index])
            target = self.coco.loadAnns(ann_ids)
            
            for i in range(len(target)):
                if exclude_all:
                    if target[i]['category_id'] not in self.cat_to_load:
                        # unknown object found in the image, therefore, ignore the image
                        images_ids_to_exclude.append(img_ids[index])
                        break
                else:
                    if target[i]['category_id'] in self.cat_to_load:
                        images_ids_to_load.append(img_ids[index])
                        break
                    
        if exclude_all:
            images_ids_to_load = np.setdiff1d(np.array(img_ids), np.array(images_ids_to_exclude)).astype(int).tolist()

        # import pdb; pdb.set_trace()


        return images_ids_to_load 

    def load_annotations(self, ann_file, classes_to_load=None, split='65_15'):
        self.coco = COCO(ann_file)
        
        self.cat_ids = self.coco.getCatIds()

        ##todo
        #G
        self.img_ids = self._filter_classes(classes_to_load, exclude_all=(False if self.test_mode else True), split=split)
        #C
        # self.img_ids = self._filter_classes(classes_to_load, exclude_all=(True if self.test_mode else True),
        #                                     split=split)
        ##
        # self.img_ids = self._filter_classes(classes_to_load, exclude_all=False, split=split)
        # self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        
        # img_infos = np.random.permutation(img_infos)[:1000]
        # self.img_ids = [img_info['id'] for img_info in img_infos]

        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # logger = get_root_logger('INFO')
        print(f"total training samples {len(img_infos)} ....")
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        # ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.cat_to_load)
        ann_info = self.coco.loadAnns(ann_ids)

        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        
        # print(f"len of valid images {len(valid_inds)}")

        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        labels_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                labels_ignore.append(self.cat2label[ann['category_id']])
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            labels_ignore = np.zeros((0, ))

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            labels_ignore=labels_ignore.astype(np.int64),
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
