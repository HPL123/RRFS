from .registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv

##
@DATASETS.register_module
class VOCDataset(XMLDataset):
    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'cat', 'chair', 'cow', 'diningtable', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'tvmonitor',
               'car', 'dog', 'sofa', 'train')

    # CLASSES = ('car', 'dog', 'sofa', 'train')

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

    def load_annotations(self, ann_file, classes_to_load=None, split=None):
        self.set_classes_split()
        img_infos = []
        classes_to_exclude = None
        if classes_to_load is None or classes_to_load == 'all':
            classes_to_exclude = None
        elif 'unseen' in classes_to_load:
            classes_to_exclude = self.seen_classes
        elif 'seen' in classes_to_load:
            classes_to_exclude = self.unseen_classes

        classes_loaded = []
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

            ##todo
            #C setting(ori code)
            include_image = True
            if classes_to_exclude is not None:
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in classes_to_exclude:
                        include_image = False
                        break
                    classes_loaded.append(name)

            if include_image == True:
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
            ##G setting
            # include_image = False
            # classes_to_load_G_setting = self.unseen_classes
            # if classes_to_exclude is not None:
            #     for obj in root.findall('object'):
            #         name = obj.find('name').text
            #         if name in classes_to_load_G_setting:
            #             include_image = True
            #             break
            #         classes_loaded.append(name)
            #
            # if include_image == True:
            #     img_infos.append(
            #         dict(id=img_id, filename=filename, width=width, height=height))
            #########

        # import pdb; pdb.set_trace()
        # files = ["VOC2007/"+filename['filename'] for filename in img_infos]
        print(f"classes loaded {np.unique(np.array(classes_loaded))}")

        return img_infos
