import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA

class ClsModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModel, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        x = self.lsm(x)
        return x
    
class ClsModelTrain(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModelTrain, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        return x

class ClsUnseen(torch.nn.Module):
    def __init__(self, att):
        super(ClsUnseen, self).__init__()
        self.W = att.type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=1024, out_features=300, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)

        print(f"__init__ {self.W.shape}")

    def forward(self, feats=None, classifier_only=False):
        f = self.fc1(feats)
        x = f.mm(self.W.transpose(1,0))
        x = self.lsm(x)

        return x

class ClsUnseenTrain(torch.nn.Module):
    def __init__(self, att):
        super(ClsUnseenTrain, self).__init__()
        self.W = att.type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=1024, out_features=300, bias=True)

        print(f"__init__ {self.W.shape}")

    def forward(self, feats=None, classifier_only=False):
        f = self.fc1(feats)
        x = f.mm(self.W.transpose(1,0))

        return x