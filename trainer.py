from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att
from mmdetection.splits import get_unseen_class_labels

opt = parse_args()


try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)

torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

unseen_attributes, unseen_att_labels = load_unseen_att(opt)
attributes, _ = load_all_att(opt)
# init classifier
trainCls = TrainCls(opt)

print('initializing GAN Trainer')


start_epoch = 0



seenDataset = FeaturesGAN(opt)
trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')

if opt.netD and opt.netG:
    start_epoch = trainFGGAN.load_checkpoint()
    
for epoch in range(start_epoch, opt.nepoch):
    # features, labels = seenDataset.epochData(include_bg=False)
    features, labels = seenDataset.epochData(include_bg=True)
    # train GAN
    trainFGGAN(epoch, features, labels)
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)
    num_of_bg = opt.syn_num*2

    real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)

    # concatenate synthesized + real bg features
    syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
    syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))
    
    trainCls(syn_feature, syn_label, gan_epoch=epoch)

    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    classes = np.concatenate((['background'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')
    # trainFGGAN.save_each_epoch_checkpoint(state=epoch)