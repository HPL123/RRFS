import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', help='coco, voc')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--class_embedding', default='VOC/fasttext_multilabels_4.npy')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train GAN')
    parser.add_argument('--nepoch_cls', type=int, default=2000, help='number of epochs to train CLS')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--cls_weight_unseen', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--lr_cls', type=float, default=0.0001, help='learning rate to train CLS ')
    parser.add_argument('--testsplit', default='test', help='unseen classes feats and labels paths')
    parser.add_argument('--trainsplit', default='train', help='seen classes feats and labels paths')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lz_ratio', type=float, default=1.0, help='mode seeking loss weight')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (for seen classes loss on fake features)")
    parser.add_argument('--pretrain_classifier_unseen', default='', help="path to pretrain classifier (for unseen classes loss on fake features)")
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--classes_split', default='')
    parser.add_argument('--outname', default='./checkpoints/', help='folder to output data and model checkpoints')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=81, help='number of all classes')
    parser.add_argument('--lr_step', type=int, default=30, help='number of all classes')
    parser.add_argument('--gan_epoch_budget', type=int, default=10000, help='random pick subset of features to train GAN')

    ##intra contra
    parser.add_argument('--lambda_contra', type=float, default=0.001, help='weight for contrastive loss')
    parser.add_argument('--num_negative', type=int, default=10, help='number of latent negative samples')
    parser.add_argument('--radius', type=float, default=0.000001, help='positive sample - distance threshold')
    parser.add_argument('--tau', type=float, default=0.1, help='temprature')
    # parser.add_argument('--featnorm', action='store_true', help='whether featnorm')

    ##inter contra
    parser.add_argument('--inter_temp', type=float, default=0.1, help='inter temperature')
    parser.add_argument('--inter_weight', type=float, default=0.001,
                        help='weight of the classification loss when learning G')

    opt = parser.parse_args()
    return opt