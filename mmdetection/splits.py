import numpy as np

IMAGENET_ALL_CLASSES = [
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

IMAGENET_UNSEEN_CLASSES = ['bench',
    'bow_tie',
    'burrito',
    'can_opener',
    'dishwasher',
    'electric_fan',
    'golf_ball',
    'hamster',
    'harmonica',
    'horizontal_bar',
    'iPod',
    'maraca',
    'pencil_box',
    'pineapple',
    'plate_rack',
    'ray',
    'scorpion',
    'snail',
    'swimming_trunks',
    'syringe',
    'tiger',
    'train',
    'unicycle'
]

VOC_ALL_CLASSES = np.array([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person', 
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
])

##todo
# COCO_ALL_CLASSES = np.array([
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
#     'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
#     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
#     'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
#     'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#     'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#     'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
# ])
COCO_ALL_CLASSES = np.array([
'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person',
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
])
##

# COCO_UNSEEN_CLASSES_65_15 = np.array([
#     'airplane', 'train', 'parking_meter', 'cat', 'bear',
#     'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich',
#     'hot_dog', 'toilet', 'mouse', 'toaster', 'hair_drier'
# ])
COCO_UNSEEN_CLASSES_65_15 = np.array([
'car', 'dog', 'sofa', 'train'
])

VOC_UNSEEN_CLASSES = np.array(['car', 'dog', 'sofa', 'train'])

COCO_SEEN_CLASSES_48_17 = np.array([
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra",
])

COCO_UNSEEN_CLASSES_48_17 = np.array([
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
])

def get_class_labels(dataset):
    if dataset == 'coco':
        return COCO_ALL_CLASSES
    elif dataset == 'voc': 
        return VOC_ALL_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_ALL_CLASSES

##todo
def get_unseen_class_labels(dataset, split='65_15'):
    if dataset == 'coco':
        # return COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
        return COCO_UNSEEN_CLASSES_65_15
    elif dataset == 'voc': 
        return VOC_UNSEEN_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_UNSEEN_CLASSES
##
def get_unseen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_unseen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_unseen_voc_ids()
    elif dataset == 'imagenet':
        return get_unseen_imagenet_ids()

def get_seen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_seen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_seen_voc_ids()
    elif dataset == 'imagenet':
        return get_seen_imagenet_ids()
##todo
# def get_unseen_coco_cat_ids(split='65_15'):
#     UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
#     ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
#     return ids
def get_unseen_coco_cat_ids(split='65_15'):
    UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15
    ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0] + 1
    return ids
##
##todo
# def get_seen_coco_cat_ids(split='65_15'):
#
#     seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15) if split=='65_15' else COCO_SEEN_CLASSES_48_17
#     ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0] + 1
#     return ids
def get_seen_coco_cat_ids(split='65_15'):

    seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15)
    ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0] + 1
    return ids
##

def get_unseen_voc_ids():
    ids = np.where(np.isin(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES))[0] + 1
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_seen_voc_ids():
    seen_classes = np.setdiff1d(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES)
    ids = np.where(np.isin(VOC_ALL_CLASSES, seen_classes))[0] + 1
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_unseen_imagenet_ids():
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_imagenet_ids():
    seen_classes = np.setdiff1d(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES)
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, seen_classes))[0] + 1
    return ids
