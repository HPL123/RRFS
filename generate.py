import numpy as np
import torch
from numpy import linalg as LA

from mmdetection.splits import get_unseen_class_ids ,get_seen_class_ids


def load_all_att(opt):
    attribute = np.load(opt.class_embedding)
    labels = np.arange(len(attribute))
    attribute/=LA.norm(attribute, ord=2)
    return torch.from_numpy(attribute), torch.from_numpy(labels)

def load_seen_att(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = np.concatenate(([0], get_seen_class_ids(opt.dataset, split=opt.classes_split)))
    return attribute[classes_ids], labels[classes_ids]

def load_unseen_att_with_bg(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = np.concatenate(([0], get_unseen_class_ids(opt.dataset, split=opt.classes_split)))
    return attribute[classes_ids], labels[classes_ids]

def load_unseen_att(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = get_unseen_class_ids(opt.dataset, split=opt.classes_split)
    return attribute[classes_ids], labels[classes_ids]

