from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import model
from cls_models import ClsModel, ClsUnseen
import torch.nn.functional as F
import torch.nn as nn
import random
import losses



class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, seen_feats_mean, gen_type='FG'):

        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.unseen_classifier = ClsUnseen(unseenAtt)
        self.unseen_classifier.cuda()

        self.unseen_classifier = loadUnseenWeights(opt.pretrain_classifier_unseen, self.unseen_classifier)
        self.classifier = ClsModel(num_classes=opt.nclass_all)
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier)

        for p in self.classifier.parameters():
            p.requires_grad = False

        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        self.netG = model.MLP_G(self.opt)
        self.netD = model.MLP_CRITIC(self.opt)
        ##add
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.opt.featnorm = True
        self.inter_contras_criterion = losses.SupConLoss_clear(self.opt.inter_temp)
        ##


        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        print('\n\n#############################################################\n')
        print(self.netG, '\n')
        print(self.netD)
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        if self.opt.cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()
            self.cross_entropy_loss.cuda()
            self.inter_contras_criterion.cuda()

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        self.trainEpoch()

    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch

    ##todo
    def load_pretrain_checkpoint(self):
        checkpoint = torch.load(self.opt.pretrain_GAN_netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        self.netD.load_state_dict(torch.load(self.opt.pretrain_GAN_netD)['state_dict'])
        print(f"loaded weights from best GAN model")

    ##

    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    ##todo
    def save_each_epoch_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch},
                   f'{self.opt.outname}/gen_{state}.pth')
    ##

    def generate_syn_feature(self, labels, attribute, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features
            2) labels of synthesised  features
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num, self.opt.resSize)
        syn_label = torch.LongTensor(nclass * num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)

        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))

                    syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i * num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))

                syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i * num, num).fill_(label)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        if 'BG' == self.gen_type:
            batch_label *= 0
        return batch_feature, batch_label, batch_att

   
    def calc_gradient_penalty(self, real_data, fake_data, input_att, contra=False):
        if contra:
            alpha = torch.rand(real_data.size(0), 1)
        else:
            alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    #############################
    def get_z_random(self):
        """
        returns normal initialized noise tensor
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def compute_contrastive_loss(self, feat_q, feat_k):
        # feat_q = F.softmax(feat_q, dim=1)
        # feat_k = F.softmax(feat_k, dim=1)
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

    def latent_augmented_sampling(self):
        query = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
        pos = torch.cuda.FloatTensor(query.shape).uniform_(-self.opt.radius, self.opt.radius).add_(query)
        negs = []
        for k in range(self.opt.num_negative):
            neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
            while (neg - query).abs().min() < self.opt.radius:
                neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz, 'gauss')
            negs.append(neg)
        return query, pos, negs

    def get_z_random_v2(self, batchSize, nz, random_type='gauss'):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z

    def trainEpoch(self):

        for i in range(0, self.ntrain, self.opt.batch_size):
            # import pdb; pdb.set_trace()
            input_res, input_label, input_att = self.sample()

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, input_att = input_res.type(torch.FloatTensor).cuda(), input_label.type(
                torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG

                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(self.mone)

                ##real inter contra loss
                input_res_norm = F.normalize((input_resv), dim=1)
                real_inter_contras_loss = self.inter_contras_criterion(input_res_norm, input_label)
                real_inter_contras_loss = real_inter_contras_loss.requires_grad_()
                real_inter_contras_loss.backward()


                ##
                z_random = self.get_z_random()
                query, pos, negs = self.latent_augmented_sampling()
                z_random2 = [query, pos] + negs

                z_conc = torch.cat([z_random] + z_random2, 0)
                label_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)

                fake = self.netG(z_conc, label_conc)
                fake1 = fake[:input_resv.size(0)]
                fake2 = fake[input_resv.size(0):]
                ##

                criticD_fake = self.netD(fake1.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(self.one)

                # gradient penalty
                ##todo
                # gradient_penalty = self.calc_gradient_penalty(input_res, fake1.data, input_att)
                gradient_penalty = self.calc_gradient_penalty(input_res, fake1.data, input_att, contra=False)
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # D_cost.backward()

                criticD_real2 = self.netD(input_resv, input_attv)
                criticD_real2 = criticD_real2.mean()
                criticD_real2.backward(self.mone)

                criticD_fake2 = self.netD(fake2.detach(), input_attv.repeat(self.opt.num_negative + 2, 1))
                ##todo
                # criticD_fake2 = criticD_fake2[:input_resv.size(0)]
                ##
                criticD_fake2 = criticD_fake2.mean()
                criticD_fake2.backward(self.one)

                # gradient penalty
                ##todo
                # gradient_penalty2 = self.calc_gradient_penalty(input_res, fake2.data[:input_resv.size(0)], input_att)
                gradient_penalty2 = self.calc_gradient_penalty(input_res.repeat(self.opt.num_negative + 2, 1),
                                                               fake2.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),
                                                               contra=True)
                ##
                gradient_penalty2.backward()

                Wasserstein_D2 = criticD_real2 - criticD_fake2
                D_cost2 = criticD_fake2 - criticD_real2 + gradient_penalty2
                # D_cost2.backward()

                self.optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            self.netG.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            ##
            z_random = self.get_z_random()
            query, pos, negs = self.latent_augmented_sampling()
            z_random2 = [query, pos] + negs

            z_conc = torch.cat([z_random] + z_random2, 0)
            label_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)

            fake = self.netG(z_conc, label_conc)
            fake1 = fake[:input_resv.size(0)]
            fake2 = fake[input_resv.size(0):]
            ##

            criticG_fake = self.netD(fake1, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake

            ##todo
            criticG_fake2 = self.netD(fake2[:input_resv.size(0)], input_attv)
            # criticG_fake2 = self.netD(fake2, input_attv.repeat(self.opt.num_negative+2, 1))
            ##
            criticG_fake2 = criticG_fake2.mean()
            G_cost2 = criticG_fake2

            ##inter contra loss
            input_res_norm_2 = F.normalize((input_resv), dim=1)
            fake_res1 = F.normalize((fake1), dim=1)
            fake_res2 = F.normalize((fake2[:input_resv.size(0)]), dim=1)

            all_features = torch.cat((fake_res1, fake_res2, input_res_norm_2.detach()), dim=0)
            fake_inter_contras_loss = self.inter_contras_criterion(all_features,
                                                                   torch.cat((input_label, input_label, input_label),
                                                                             dim=0))
            # fake_inter_contras_loss.requires_grad_()
            fake_inter_contras_loss = self.opt.inter_weight * fake_inter_contras_loss
            # fake_inter_contras_loss.backward(retain_graph=True)
            #####################################################################

            self.loss_contra = 0.0
            for j in range(input_res.size(0)):
                logits = fake2[j:fake2.shape[0]:input_res.size(0)].view(self.opt.num_negative + 2, -1)

                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)

                self.loss_contra += self.compute_contrastive_loss(logits[0:1], logits[1:])

            loss_lz = self.opt.lambda_contra * self.loss_contra

            # ---------------------
            # classification loss
            ##
            c_errG = self.cls_criterion(self.classifier(feats=fake1, classifier_only=True), Variable(input_label))

            c_errG = self.opt.cls_weight * c_errG
            # --------------------------------------------

            errG = -G_cost - G_cost2 + c_errG + loss_lz + fake_inter_contras_loss


            errG.backward()
            self.optimizerG.step()

            
            print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
                        Loss: {errG.item() :0.4f} D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} \
                        seen loss: {c_errG.data.item():.4f} loss div: {loss_lz.item():0.4f} real_inter_contras_loss: {real_inter_contras_loss.data.item():.4f} fake_inter_contras_loss : {fake_inter_contras_loss.data.item():.4f}")
        self.netG.eval()
