from .registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
from splits import get_unseen_class_ids, get_seen_class_ids

@DATASETS.register_module
class ILSVRCDataset(XMLDataset):

    
    CLASSES = [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]
    
    # np.array(map_det.values()) == np.array(self.CLASSES)
    def __init__(self, **kwargs):
        super(ILSVRCDataset, self).__init__(**kwargs)
        # self.cat_to_load = self.cat2label.values()
        # print(self.cat_to_load)
    #     # if 'VOC2007' in self.img_prefix:
    #     #     self.year = 2007
    #     # elif 'VOC2012' in self.img_prefix:
    #     #     self.year = 2012
    #     # else:
    #     #     raise ValueError('Cannot infer dataset year from img_prefix')
    
    def load_annotations(self, ann_file,  classes_to_load=None, split=None):
        img_infos = []
        unseen_class_ids, seen_class_ids = get_unseen_class_ids('imagenet')-1, get_seen_class_ids('imagenet')-1

        if self.classes_to_load == 'seen':
            self.cat_to_load = seen_class_ids
        elif self.classes_to_load == 'unseen':
            self.cat_to_load = unseen_class_ids

        self.class_names_to_load = np.array(self.CLASSES)[self.cat_to_load]
        print(self.class_names_to_load)
        
        img_ids = mmcv.list_from_file(ann_file)
        img_ids = [img_id.split(' ')[0] for img_id in img_ids]
        img_ids = [img_id for img_id in img_ids if 'extra' not in img_id]
        map_det_file = open('map_det.txt', 'r')
        map_det = map_det_file.readlines()
        map_det = [det.strip().split(' ') for det in map_det]
        self.map_det = {det[0]: det[2] for det in map_det}

        for img_id in img_ids:
            data_split = 'train' if 'train' in img_id else 'val'

            filename = f'Data/DET/{data_split}/{img_id}.JPEG'
            xml_path = f'{self.img_prefix}Annotations/DET/{data_split}/{img_id}.xml'

            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            include_image = self.should_include_image(root)
            
            if include_image == True:
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
        print(f'total {self.classes_to_load} images loaded {len(img_infos)}')
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
                name = self.map_det[obj.find('name').text]
                if name in self.class_names_to_load:
                    include_image = True
                else:
                    include_image = False
                    # print(f"{name} in image")
                break
        else:
            for obj in root.findall('object'):
                name = self.map_det[obj.find('name').text]
                if name in self.class_names_to_load:
                    include_image = True
                    break
                # else:
                #     print(f"{name} in image")

        return include_image

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        data_split = 'train' if 'train' in img_id else 'val'
        xml_path = f'{self.img_prefix}Annotations/DET/{data_split}/{img_id}.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = self.map_det[obj.find('name').text]
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
