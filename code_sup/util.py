import numpy as np
import torch

from sklearn.metrics import confusion_matrix
from mmdetection.splits import get_unseen_class_labels

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every lr_step epochs"""
    lr = opt.lr_cls * (0.1 ** (epoch //  opt.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def val(dataloader, classifier, criterion, opt, epoch, verbose="Val"):
    preds_all = []
    gt_all = []
    loss_epoch = 0
    classifier.eval()
    for ite, (in_feat, in_label)  in enumerate(dataloader):

        in_feat = in_feat.cuda()
        in_label = in_label.cuda()
        preds = classifier(feats=in_feat, classifier_only=True)

        preds_all.append(preds.data.cpu().numpy())
        gt_all.append(in_label.data.cpu().numpy())

        loss = criterion(preds, in_label)
        loss_epoch+=loss.item()
        if ite % 100 == 99:
            print(f'{verbose}   Epoch [{epoch+1:02}/{opt.nepoch_cls}] Iter [{ite:05}/{len(dataloader)}]{ite/len(dataloader) * 100:02.3f}% Loss: {loss_epoch/ite :0.4f}')
 
    preds_all = np.concatenate(preds_all)
    gt_all = np.concatenate(gt_all)
    return compute_per_class_acc(gt_all, preds_all, opt, verbose=verbose)

def compute_per_class_acc(test_label, predicted_label, opt, verbose="Val"):
    class_labels = np.unique(test_label)
    acc_per_class = torch.FloatTensor(len(class_labels)).fill_(0)
    predicted_label = torch.max(torch.from_numpy(predicted_label), 1)[1]
    test_label = torch.from_numpy(test_label)

    classes = np.concatenate((['background'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))

    for index, label in enumerate(class_labels):
        idx = (test_label == label)
        acc_per_class[index] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        print(f"[{verbose}] {classes[label]}: {acc_per_class[index]:0.4f}")

    c_mat = confusion_matrix(test_label, predicted_label)
    
    acc = acc_per_class.mean()
    print(f"\n------------------------\n[{verbose}] Mean: {acc:0.4f} \n")

    return acc, acc_per_class, c_mat

def loadUnseenWeights(file_path, model):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    return model

def loadFasterRcnnCLSHead(filepath, model):
    checkpoint = torch.load(filepath)
    state_dict = checkpoint['state_dict']
    own_dict = model.state_dict()
    own_dict['fc1.weight'].copy_(state_dict['bbox_head.fc_cls.weight'].cuda())
    own_dict['fc1.bias'].copy_(state_dict['bbox_head.fc_cls.bias'].cuda())

    assert (model.state_dict()['fc1.bias'] == state_dict['bbox_head.fc_cls.bias'].cuda()).all(), 'Something wrong with loading pretrained fasterrcnn cls head!!'
    print(f"loaded classifier from {filepath}")
    return model

# for i in $(la /raid/mun/codes/zero_shot_detection/zsd_copy_2/checkpoints/VOC_new_unseen_cls_10/classifier_best*); do $(./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 6 --out voc_results.pkl --syn_weights $i &>> results.out); done

# for i in $(la /raid/mun/codes/zero_shot_detection/zsd_copy_2/checkpoints/VOC_new_unseen_cls_10/latest_10/classifier_latest*); do $(./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 6 --out voc_results.pkl --syn_weights $i &>> latest_results.out); done

