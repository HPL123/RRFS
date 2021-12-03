import numpy as np
import torch
# import pandas as pd 
from torch.utils.data import Dataset
from util import *

import os.path
from os import path

class FeaturesCls(Dataset):
     
    def __init__(self, opt, features=None, labels=None, val=False, split='seen', classes_to_train=None):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        self.classes_to_train = classes_to_train
        self.classid_tolabels = None
        self.features = features
        self.labels = labels
        if self.classes_to_train is not None:
            self.classid_tolabels = {label: i for i, label in enumerate(self.classes_to_train)}
        
        print(f"class ids for unseen classifier {self.classes_to_train}")
        if 'test' in split:
            self.loadRealFeats(syn_feature=features, syn_label=labels, split=split)

    def loadRealFeats(self, syn_feature=None, syn_label=None, split='train'):
        if 'test' in split:
            self.features = np.load(f"{self.root}/{self.opt.testsplit}_feats.npy")
            self.labels = np.load(f"{self.root}/{self.opt.testsplit}_labels.npy")
            print(f"{len(self.labels)} testsubset {self.opt.testsplit} features loaded")
            # import pdb; pdb.set_trace()
    
    def replace(self, features=None, labels=None):
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        print(f"\n=== Replaced new batch of Syn Feats === \n")

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)

class FeaturesGAN():
    def __init__(self, opt):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        # self.attribute = np.load(opt.class_embedding)

        print("loading numpy arrays")
        self.all_features = np.load(f"{self.root}/{self.opt.trainsplit}_feats.npy")
        self.all_labels = np.load(f"{self.root}/{self.opt.trainsplit}_labels.npy")
        mean_path = f"{self.root}/{self.opt.trainsplit}_mean.npy"
        print(f'loaded data from {self.opt.trainsplit}')
        self.pos_inds = np.where(self.all_labels>0)[0]
        self.neg_inds = np.where(self.all_labels==0)[0]

        unique_labels = np.unique(self.all_labels)
        self.num_bg_to_take = len(self.pos_inds)//len(unique_labels)

        print(f"loaded {len(self.pos_inds)} fg labels")
        print(f"loaded {len(self.neg_inds)} bg labels ")
        print(f"bg indexes for each epoch {self.num_bg_to_take}")


        self.features_mean = np.zeros((max(unique_labels) + 1 , self.all_features.shape[1]))
        # if path.exists(mean_path):
        #     self.features_mean = np.load(mean_path)
        # else:
        #     for label in unique_labels:
        #         label_inds = np.where(self.all_labels==label)[0]
        #         self.features_mean[label] = self.all_features[label_inds].mean(axis=0)
        #     np.save(mean_path, self.features_mean)
        




    def epochData(self, include_bg=False):
        fg_inds = np.random.permutation(self.pos_inds)
        inds = np.random.permutation(fg_inds)[:int(self.opt.gan_epoch_budget)]
        if include_bg:
            bg_inds = np.random.permutation(self.neg_inds)[:self.num_bg_to_take]
            inds = np.random.permutation(np.concatenate((fg_inds, bg_inds)))[:int(self.opt.gan_epoch_budget)]
        features = self.all_features[inds]
        labels = self.all_labels[inds]    
        return features, labels

    def getBGfeats(self, num=1000):
        bg_inds = np.random.permutation(self.neg_inds)[:num]
        print(f"{len(bg_inds)} ")
        return self.all_features[bg_inds], self.all_labels[bg_inds]
    def __len__(self):
        return len(self.all_labels)


