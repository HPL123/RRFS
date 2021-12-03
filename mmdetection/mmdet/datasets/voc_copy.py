from .registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
from splits import get_unseen_class_ids, get_seen_class_ids

@DATASETS.register_module
class VOCDataset(XMLDataset):
    
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'cat', 'chair', 'cow', 'diningtable', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'tvmonitor',
               'car', 'dog', 'sofa', 'train')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
    
    def set_classes_split(self):
        self.unseen_classes = ['car', 'dog', 'sofa', 'train']
        self.seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'cat', 'chair', 'cow', 'diningtable', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor']

    def load_annotations(self, ann_file,  classes_to_load=None, split=None):
        self.set_classes_split()
        unseen_class_ids, seen_class_ids = get_unseen_class_ids('voc')-1, get_seen_class_ids('voc')-1

        img_infos = []
        if self.classes_to_load == 'seen':
            self.cat_to_load = seen_class_ids
        elif self.classes_to_load == 'unseen':
            self.cat_to_load = unseen_class_ids

        self.class_names_to_load = np.array(self.CLASSES)[self.cat_to_load]

        # classes_loaded = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # include_image = include_image = self.should_include_image(root)
            include_image = self.should_include_image(root)
            # if classes_to_exclude is not None:
            #     for obj in root.findall('object'):
            #         name = obj.find('name').text
            #         if name in classes_to_exclude:
            #             include_image = False
            #             break
                    # classes_loaded.append(name)

            if include_image == True:
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
        
        # import pdb; pdb.set_trace()
        # files = ["VOC2007/"+filename['filename'] for filename in img_infos]
        # print(f"classes loaded {np.unique(np.array(classes_loaded))}")

        return img_infos

    def should_include_image(self, root):
        """
        root: xml file parser
        while loading annotations checks whether to include image in the dataset
        checks for each obj name in the class_names_to_load list
        for seen classes we strictly exclude objects if an unseen object is present
        for unseen classes during validation we load the image if the unseen object is present and ignore the annotation for seen object
        """
        include_image = False
        if self.classes_to_load == 'seen':
            # include stricktly only images with seen objects 
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.class_names_to_load:
                    include_image = True
                else:
                    include_image = False
                    break
        else:
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.class_names_to_load:
                    include_image = True
                    break
        return include_image
    
    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
        # xml_path = f'{self.img_prefix}Annotations/DET/{self.data_split}/{img_id}.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_names_to_load:
                # ignore the annotations for the object
                continue
            label = self.cat2label[name]
            difficult = 0
            # difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
