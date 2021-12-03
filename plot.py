# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from util import *
import seaborn as sns
from mmdetection.splits import * 

colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45',
'#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3', '#5b7cd4',
'#3e319d', '#a837b2', '#400dd2', '#f8d307']
# losses_plot
def plot_gan_losses(Losses_D, Losses_G, W_dist, opt, prefix='att'):
    
    x = np.linspace(0, len(Losses_D), len(Losses_D))
    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, Losses_D, "-r", label="Loss D")
    plt.plot(x, Losses_G, "-g", label="Loss G")
    plt.plot(x, W_dist, "-b", label="W dist")

    plt.legend(loc="upper left")
    fig.savefig(f'{opt.outname}/{prefix}_gan_losses.pdf', format='pdf', dpi=600)
    plt.close()

def plot_acc(acc, opt, prefix='att'):
    
    x = np.linspace(0, len(acc), len(acc)) / opt.nepoch_cls
    fig = plt.figure(figsize=(10, 10))
   
    classes = np.concatenate((['bg'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))
    # import pdb; pdb.set_trace()
   
    for i in range(acc.shape[1]):
        plt.plot(x, acc[:, i], colors[i], linewidth=1)#label=classes[i], 
        # plt.plot(x, np.ones(shape=acc.shape[0])*max(acc[:, i]), colors[i])
        plt.text(x = max(x)+0.01 , y = acc[-1, i], s = f"{classes[i]} {100*(acc[-1, i]):02.2f}", color=colors[i])
        plt.text(x = x[np.argmax(acc[:, i])] , y = np.max(acc[:, i])+0.01, s = f"{int(100*max(acc[:, i])):02}", color=colors[i])
        plt.plot(x[np.argmax(acc[:, i])], np.max(acc[:, i]), 'bo')

    plt.plot(x, acc.mean(1), colors[i+1], linewidth=3)# label="mean",
    # plt.plot(x, np.ones(shape=acc.shape[0])*max(acc.mean(1)), colors[i+1])
    plt.text(x = max(x)+0.01 , y = acc.mean(1)[-1], s = f"mean  {(100*(acc.mean(1)[-1])):02.2f}", color=colors[i+1])
    plt.text(x = x[np.argmax(acc.mean(1))] , y = np.max(acc.mean(1))+0.01, s = f"{100* max(acc.mean(1)):02.2f}" , color=colors[i+1])
    plt.plot(x[np.argmax(acc.mean(1))], np.max(acc.mean(1)), 'bo') 

    sns.despine(top=True, right=True, left=False, bottom=False)


    plt.legend(loc="lower right", frameon=False)
    fig.savefig(f'{opt.outname}/{prefix}_classifier_acc.pdf', format='pdf', dpi=600)
    plt.close()
   
# plot_gan_losses(np.load('results/Losses_D.npy'),  np.load('results/Losses_G.npy'), np.load('results/W_dist.npy'))
# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

# plot_acc(np.load('results/val_accuracies.npy'), dotdict({'nepoch_cls': 20, 'dataset': 'coco',  'outname': 'checkpoints/coco'}))

# labels = get_class_labels('voc')

def plot_confusion_matrix(c_mat, xtick_marks, ytick_marks, opt, dataset='val', prefix='att'):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(c_mat, cmap=plt.cm.Blues)

    plt.xticks(np.arange(xtick_marks.shape[0]), xtick_marks, rotation=75)
    plt.yticks(np.arange(ytick_marks.shape[0]), ytick_marks)
    # import pdb; pdb.set_trace()

    for (i, j), z in np.ndenumerate(c_mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    fig = plt.gcf()
    fig.savefig(f'{opt.outname}/{prefix}_confusion_matrix_{dataset}.pdf', format='pdf', dpi=600)
    plt.close()
# c_mat = np.load(f'confusion_matrix_Train.npy')
# classes = np.concatenate((['background'], ['car', 'dog', 'sofa', 'train']))
# plot_confusion_matrix(c_mat, classes, classes, dataset='Train')
