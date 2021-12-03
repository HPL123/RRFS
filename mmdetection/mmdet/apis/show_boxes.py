# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from splits import COCO_ALL_CLASSES, get_unseen_class_labels
# color_map = {label: (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)) for label in COCO_ALL_CLASSES}

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)
    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='black',
                      thickness=4,
                      font_scale=1.1,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      dataset=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """

    # import random
    # color = "%06x" % random.randint(0, 0xFFFFFF)
    # from splits import COCO_ALL_CLASSES
    # color_map = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in COCO_ALL_CLASSES}

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    ##todo
    # img resize
    unseen_labels = get_unseen_class_labels(dataset, '177_23')
    # unseen_labels = ['car', 'dog', 'sofa', 'train'
    #                  ]

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    
    # import pdb; pdb.set_trace()

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        
        if label_text in unseen_labels:
            bbox_color = color_val('red')
            # print(label_text)
        else:
            bbox_color = color_val('green')


        print(bbox_color, label_text)
        # bbox_color = color_map[label_text] if color_map is not None else bbox_color
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        
        label_text = label_text.title().replace('_', ' ')
        # if len(bbox) > 4:
        #     label_text += ': {}'.format(int(100*bbox[-1]))
        
        # import pdb; pdb.set_trace()
        txt_thickness = 4
        font = cv2.FONT_HERSHEY_SIMPLEX 
        (text_width, text_height) = cv2.getTextSize(label_text, font, fontScale=font_scale, thickness=txt_thickness)[0]
        text_offset_x = bbox_int[0]
        text_offset_y = bbox_int[1] - 2
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        rectangle_bgr = (255, 255, 255)
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    font, font_scale, text_color, thickness=txt_thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)