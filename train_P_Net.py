import os
import sys
import datetime
import time
import numpy as np

import sklearn.metrics as metrics

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from dataloader.OCT_DataLoader import OCT_ClsDataloader
from dataloader.fundus_cls_dataloader import NewClsFundusDataloader
from networks.P_Net_v1 import Strcutre_Extraction_Network, Image_Reconstruction_Network
from networks.discriminator import Discriminator
from utils.gan_loss import AdversarialLoss
from utils.visualizer import Visualizer
from utils.utils import adjust_lr, cuda_visible, print_args, save_ckpt, AverageMeter, LastAvgMeter
from utils.parser import ParserArgs


class PNetModel(nn.Module):
    def __init__(self, args, ablation_mode=4):
        super(PNetModel, self).__init__()
        self.args = args

        """
        ablation study mode
        """
        # 0: output_structure                       (1 feature)
        # 2: image (1 feature), i.e. auto-encoder
        # 4: output_structure + image               (2 features)

        # model on gpu
        if self.args.data_modality == 'fundus':
            model_G1 = Strcutre_Extraction_Network(n_channels=1, n_classes=1)
        else:
            model_G1 = Strcutre_Extraction_Network(n_channels=1, n_classes=12)
        model_G2 = Image_Reconstruction_Network(in_ch=1, modality=self.args.data_modality, ablation_mode=ablation_mode)
        model_D = Discriminator(in_channels=1)

        model_G1 = nn.DataParallel(model_G1).cuda()
        model_G2 = nn.DataParallel(model_G2).cuda()
        model_D = nn.DataParallel(model_D).cuda()

        l1_loss = nn.L1Loss().cuda()
        adversarial_loss = AdversarialLoss().cuda()

        self.add_module('model_G1', model_G1)
        self.add_module('model_G2', model_G2)
        self.add_module('model_D', model_D)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # optimizer
        self.optimizer_G = torch.optim.Adam(params=self.model_G2.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr * args.d2g_lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        # load 1-st ckpts
        if self.args.server == 'ai':
            seg_ckpt_root = os.path.join('/root/workspace', args.project, 'save_models')
        else:
            seg_ckpt_root = os.path.join('/home/imed/new_disk/workspace', args.project, 'save_models')
        if self.args.data_modality == 'fundus':
            if self.args.DA_ablation_mode_isee == 0:
                _g_zero_point = '0'
            elif self.args.DA_ablation_mode_isee == 0.001:
                _g_zero_point = '001'
            elif self.args.DA_ablation_mode_isee == 0.0001:
                # this is the default
                _g_zero_point = '0001'
            else:
                raise NotImplementedError('error')

            seg_ckpt_path = os.path.join(seg_ckpt_root, '1st_fundus_seg_g_{}.pth.tar'.format(_g_zero_point))

            ## orginal seg mdel
            # seg_ckpt_path = os.path.join(seg_ckpt_root, '1st_fundus_seg_vgg.pth.tar')
        else:
            seg_ckpt_path = os.path.join(seg_ckpt_root, '1st_oct_seg.pth.tar')

        if os.path.isfile(seg_ckpt_path):
            print("=> loading G1 checkpoint")
            checkpoint = torch.load(seg_ckpt_path)
            self.model_G1.load_state_dict(checkpoint['state_dict_G'])
            print("=> loaded G1 checkpoint (epoch {}) \n from {}"
                  .format(checkpoint['epoch'], seg_ckpt_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(seg_ckpt_path))

        # Optionally resume from a checkpoint
        if self.args.resume:
            ckpt_root = os.path.join(self.args.output_root, args.project, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading G2 checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.model_G2.load_state_dict(checkpoint['state_dict_G'])
                print("=> loaded G2 checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def process(self, image):
        # process_outputs
        seg_mask, image_rec = self(image)

        """
        G and D process, this package is reusable
        """
        # zero optimizers
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        gen_loss = 0
        dis_loss = 0

        real_B = image
        fake_B = image_rec

        # discriminator loss
        dis_input_real = real_B
        dis_input_fake = fake_B.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = fake_B
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.args.lamd_gen
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.args.lamd_fm
        gen_loss += gen_fm_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(fake_B, real_B) * self.args.lamd_p
        gen_loss += gen_l1_loss

        # create logs
        logs = dict(
            gen_gan_loss=gen_gan_loss,
            gen_fm_loss=gen_fm_loss,
            gen_l1_loss=gen_l1_loss,
            # gen_content_loss=gen_content_loss,
            # gen_style_loss=gen_style_loss,
        )

        return seg_mask, fake_B, gen_loss, dis_loss, logs

    def forward(self, image):
        with torch.no_grad():
            seg_mask, seg_structure_feat, seg_latent_feat = self.model_G1(image)
        image_rec = self.model_G2(image, seg_mask, seg_latent_feat)

        return seg_mask, image_rec

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.optimizer_D.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.optimizer_G.step()


class RunMyModel(object):
    def __init__(self):
        args = ParserArgs().get_args()
        cuda_visible(args.gpu)

        cudnn.benchmark = True

        self.vis = Visualizer(env='{}'.format(args.version), port=args.port, server=args.vis_server)

        if args.data_modality == 'fundus':
            # IDRiD dataset for segmentation
            # image, mask, image_name_item

            # iSee dataset for classification
            # image, image_name
            self.train_loader, self.normal_test_loader, \
            self.amd_fundus_loader, self.myopia_fundus_loader, \
            self.glaucoma_fundus_loader, self.dr_fundus_loader = \
                NewClsFundusDataloader(data_root=self.args.isee_fundus_root,
                                       batch=self.args.batch,
                                       scale=self.args.scale).data_load()

        else:
            # Challenge OCT dataset for classification
            # image, [case_name, image_name]
            self.train_loader, self.normal_test_loader, self.oct_abnormal_loader = OCT_ClsDataloader(
                                                    data_root=args.challenge_oct,
                                                       batch=args.batch,
                                                       scale=args.scale).data_load()

        print_args(args)
        self.args = args
        self.new_lr = self.args.lr
        self.model = PNetModel(args)

        if args.predict:
            self.test_acc()
        else:
            self.train_val()

    def train_val(self):
        # general metrics
        self.best_auc = 0
        self.is_best = False
        # self.total_auc_top10 = AverageMeter()
        self.total_auc_last10 = LastAvgMeter(length=10)
        self.acc_last10 = LastAvgMeter(length=10)

        # metrics for iSee
        self.myopia_auc_last10 = LastAvgMeter(length=10)
        self.amd_auc_last10 = LastAvgMeter(length=10)
        self.glaucoma_auc_last10 = LastAvgMeter(length=10)
        self.dr_auc_last10 = LastAvgMeter(length=10)

        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            if self.args.data_modality == 'fundus':
                # total: 1000
                adjust_lr_epoch_list = [40, 80, 160, 240]
            else:
                # total: 180
                adjust_lr_epoch_list = [20, 40, 80, 120]
            _ = adjust_lr(self.args.lr, self.model.optimizer_G, epoch, adjust_lr_epoch_list)
            new_lr = adjust_lr(self.args.lr, self.model.optimizer_D, epoch, adjust_lr_epoch_list)
            self.new_lr = min(new_lr, self.new_lr)

            self.epoch = epoch
            self.train()
            # last 80 epoch, validate with freq
            if epoch > self.args.validate_start_epoch \
                    and (epoch % self.args.validate_freq == 0
                         or epoch > (self.args.n_epochs - self.args.validate_each_epoch)):
                self.validate_cls()

            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Node: {}'.format(self.args.node))
            print('GPU: {}'.format(self.args.gpu))
            print('Version: {}\n'.format(self.args.version))

    def train(self):
        self.model.train()
        prev_time = time.time()
        train_loader = self.train_loader

        for i, (image, _,) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)

            # train
            seg_mask, image_rec, gen_loss, dis_loss, logs = \
                self.model.process(image)

            # backward
            self.model.backward(gen_loss, dis_loss)

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = self.epoch * train_loader.__len__() + i
            batches_left = self.args.n_epochs * train_loader.__len__() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                             (self.epoch, self.args.n_epochs,
                              i, train_loader.__len__(),
                              dis_loss.item(),
                              gen_loss.item(),
                              time_left))

            # --------------
            #  Visdom
            # --------------
            if i % self.args.vis_freq == 0:
                image = image[:self.args.vis_batch]

                if self.args.data_modality == 'oct':
                    # BCWH -> BWH, torch.max in Channel dimension
                    seg_mask = torch.argmax(seg_mask[:self.args.vis_batch], dim=1).float()
                    # BWH -> B1WH, 11 -> 1
                    seg_mask = (seg_mask.unsqueeze(dim=1)/11).clamp(0, 1)

                else:
                    seg_mask = seg_mask[:self.args.vis_batch].clamp(0, 1)
                image_rec = image_rec[:self.args.vis_batch].clamp(0, 1)
                image_diff = torch.abs(image-image_rec)

                vim_images = torch.cat([image, seg_mask, image_rec, image_diff], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

                output_save = os.path.join(self.args.output_root,
                                           self.args.project,
                                           'output_v1_0812',
                                           self.args.version,
                                           'train')
                if not os.path.exists(output_save):
                    os.makedirs(output_save)
                tv.utils.save_image(vim_images, os.path.join(output_save, '{}.png'.format(i)), nrow=4)

            if i+1 == train_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=dis_loss.item(),
                                             lr=self.new_lr))
                self.vis.plot_single_win(dict(gen_loss=gen_loss.item(),
                                              gen_l1_loss=logs['gen_l1_loss'].item(),
                                              gen_fm_loss=logs['gen_fm_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item(),
                                              gen_content_loss=logs['gen_content_loss'].item(),
                                              gen_style_loss=logs['gen_style_loss'].item()),
                                         win='gen_loss')

    def validate_cls(self):
        # self.model.eval()
        self.model.train()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            if self.args.data_modality == 'fundus':
                myopia_gt_list, myopia_pred_list = self.forward_cls_dataloader(
                    loader=self.myopia_fundus_loader, is_disease=True)

                amd_gt_list, amd_pred_list = self.forward_cls_dataloader(
                    loader=self.amd_fundus_loader, is_disease=True
                )
                glaucoma_gt_list, glaucoma_pred_list = self.forward_cls_dataloader(
                    loader=self.glaucoma_fundus_loader, is_disease=True
                )
                dr_gt_list, dr_pred_list = self.forward_cls_dataloader(
                    loader=self.dr_fundus_loader, is_disease=True
                )
            else:
                abnormal_gt_list, abnormal_pred_list = self.forward_cls_dataloader(
                    loader=self.oct_abnormal_loader, is_disease=True)

            _, normal_train_pred_list = self.forward_cls_dataloader(
                loader=self.train_loader, is_disease=False
            )
            normal_gt_list, normal_pred_list = self.forward_cls_dataloader(
                loader=self.normal_test_loader, is_disease=False)

            """
            computer metrics
            """
            # Difference: total_true_list and total_pred_list
            if self.args.data_modality == 'fundus':
                # test metrics for myopia
                m_true_list = myopia_gt_list + normal_gt_list
                m_pred_list = myopia_pred_list + normal_pred_list
                # test metrics for amd
                a_true_list = amd_gt_list + normal_gt_list
                a_pred_list = amd_pred_list + normal_pred_list
                # test metrics for glaucoma
                g_true_list = glaucoma_gt_list + normal_gt_list
                g_pred_list = glaucoma_pred_list + normal_pred_list
                # test metrics for amd
                d_true_list = dr_gt_list + normal_gt_list
                d_pred_list = dr_pred_list + normal_pred_list
                # total
                total_true_list = a_true_list + myopia_gt_list + glaucoma_gt_list + dr_gt_list
                total_pred_list = a_pred_list + myopia_pred_list + glaucoma_pred_list + dr_pred_list

                # fpr, tpr, thresholds = metrics.roc_curve()
                myopia_auc = metrics.roc_auc_score(np.array(m_true_list), np.array(m_pred_list))
                amd_auc = metrics.roc_auc_score(np.array(a_true_list), np.array(a_pred_list))
                glaucoma_auc = metrics.roc_auc_score(np.array(g_true_list), np.array(g_pred_list))
                dr_auc = metrics.roc_auc_score(np.array(d_true_list), np.array(d_pred_list))
            else:
                total_true_list = abnormal_gt_list + normal_gt_list
                total_pred_list = abnormal_pred_list + normal_pred_list

            # get roc curve and compute the auc
            fpr, tpr, thresholds = metrics.roc_curve(np.array(total_true_list), np.array(total_pred_list))
            total_auc = metrics.auc(fpr, tpr)

            """
            compute thereshold, and then compute the accuracy
            """
            percentage = 0.75
            _threshold_for_acc = sorted(normal_train_pred_list)[int(len(normal_train_pred_list) * percentage)]
            normal_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in normal_pred_list]
            amd_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in amd_pred_list]
            myopia_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in myopia_pred_list]
            glaucoma_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in glaucoma_pred_list]
            dr_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in dr_pred_list]

            # acc, sensitivity and specifity
            def calcu_cls_acc(pred_list, gt_list):
                cls_pred_list = normal_cls_pred_list + pred_list
                gt_list = normal_gt_list + gt_list
                acc = metrics.accuracy_score(y_true=gt_list, y_pred=cls_pred_list)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true=gt_list, y_pred=cls_pred_list).ravel()
                sen = tp / (tp + fn + 1e-7)
                spe = tn / (tn + fp + 1e-7)
                return acc, sen, spe

            total_acc, total_sen, total_spe = calcu_cls_acc(
                amd_cls_pred_list + myopia_cls_pred_list, amd_gt_list + myopia_gt_list)
            amd_acc, amd_sen, amd_spe = calcu_cls_acc(amd_cls_pred_list, amd_gt_list)
            myopia_acc, myopia_sen, myopia_spe = calcu_cls_acc(myopia_cls_pred_list, myopia_gt_list)

            # update
            if self.args.data_modality:
                self.myopia_auc_last20.update(myopia_auc)
                self.amd_auc_last20.update(amd_auc)

            self.total_auc_last20.update(total_auc)
            mean, deviation = self.total_auc_top10.top_update_calc(total_auc)

            self.is_best = total_auc > self.best_auc
            self.best_auc = max(total_auc, self.best_auc)

            """
            plot metrics curve
            """
            # ROC curve
            self.vis.draw_roc(fpr, tpr)
            # total auc, primary metrics
            self.vis.plot_single_win(dict(value=total_auc,
                                          best=self.best_auc,
                                          last_avg=self.total_auc_last20.avg,
                                          last_std=self.total_auc_last20.std,
                                          top_avg=mean,
                                          top_dev=deviation), win='total_auc')

            self.vis.plot_single_win(dict(
                total_acc=total_acc,
                total_sen=total_sen,
                total_spe=total_spe,
                amd_acc=amd_acc,
                amd_sen=amd_sen,
                amd_spe=amd_spe,
                myopia_acc=myopia_acc,
                myopia_sen=myopia_sen,
                myopia_spe=myopia_spe
            ), win='accuracy')

            # Difference
            if self.args.data_modality == 'fundus':
                self.vis.plot_single_win(dict(value=amd_auc,
                                              last_avg=self.amd_auc_last20.avg,
                                              last_std=self.amd_auc_last20.std), win='amd_auc')
                self.vis.plot_single_win(dict(value=myopia_auc,
                                              last_avg=self.myopia_auc_last20.avg,
                                              last_std=self.myopia_auc_last20.std), win='myopia_auc')

                metrics_str = 'best_auc = {:.4f},' \
                              'total_avg = {:.4f}, total_std = {:.4f}, ' \
                              'total_top_avg = {:.4f}, total_top_dev = {:.4f}, ' \
                              'amd_avg = {:.4f}, amd_std = {:.4f}, ' \
                              'myopia_avg = {:.4f}, myopia_std ={:.4f}'.format(self.best_auc,
                                       self.total_auc_last20.avg, self.total_auc_last20.std,
                                       mean, deviation,
                                       self.amd_auc_last20.avg, self.amd_auc_last20.std,
                                       self.myopia_auc_last20.avg, self.myopia_auc_last20.std)
                metrics_acc_str = '\n total_acc = {:.4f}, total_sen = {:.4f}, total_spe = {:.4f}, ' \
                                  'amd_acc = {:.4f}, amd_sen = {:.4f}, amd_spe = {:.4f}, ' \
                                  'myopia_acc = {:.4f}, myopia_sen = {:.4f}, myopia_spe = {:.4f}'\
                    .format(total_acc, total_sen, total_spe, amd_acc, amd_sen,
                            amd_spe, myopia_acc, myopia_sen, myopia_spe)

            else:
                metrics_str = 'best_auc = {:.4f},' \
                              'total_avg = {:.4f}, total_std = {:.4f}, ' \
                              'total_top_avg = {:.4f}, total_top_dev = {:.4f}'.format(self.best_auc,
                                      self.total_auc_last20.avg,
                                      self.total_auc_last20.std,
                                      mean, deviation)
                metrics_acc_str = '\n None'

            self.vis.text(metrics_str + metrics_acc_str)

        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_G': self.model.model_G2.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  is_best=self.is_best,
                  args=self.args)

        print('\n Save ckpt successfully!')
        print('\n', metrics_str + metrics_acc_str)

    def test_acc(self):
        self.model.train()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            _, normal_train_pred_list = self.forward_cls_dataloader(
                loader=self.train_loader, is_disease=False
            )

            if self.args.data_modality == 'fundus':
                myopia_gt_list, myopia_pred_list = self.forward_cls_dataloader(
                    loader=self.myopia_fundus_loader, is_disease=True)

                amd_gt_list, amd_pred_list = self.forward_cls_dataloader(
                    loader=self.amd_fundus_loader, is_disease=True
                )
            else:
                abnormal_gt_list, abnormal_pred_list = self.forward_cls_dataloader(
                    loader=self.oct_abnormal_loader, is_disease=True)

            normal_gt_list, normal_pred_list = self.forward_cls_dataloader(
                loader=self.normal_test_loader, is_disease=False)

            """
            compute metrics
            """
            # Difference: total_true_list and total_pred_list
            if self.args.data_modality == 'fundus':
                # test metrics for amd
                amd_auc_true_list = amd_gt_list + normal_gt_list
                amd_auc_pred_list = amd_pred_list + normal_pred_list
                # myopia
                myopia_auc_true_list = myopia_gt_list + normal_gt_list
                myopia_auc_pred_list = myopia_pred_list + normal_pred_list
                # total
                total_true_list = amd_auc_true_list + myopia_gt_list
                total_pred_list = amd_auc_pred_list + myopia_pred_list

                # fpr, tpr, thresholds = metrics.roc_curve()
                myopia_auc = metrics.roc_auc_score(np.array(myopia_auc_true_list), np.array(myopia_auc_pred_list))
                amd_auc = metrics.roc_auc_score(np.array(amd_auc_true_list), np.array(amd_auc_pred_list))

            else:
                total_true_list = abnormal_gt_list + normal_gt_list
                total_pred_list = abnormal_pred_list + normal_pred_list

            # get roc curve and compute the auc
            fpr, tpr, thresholds = metrics.roc_curve(np.array(total_true_list), np.array(total_pred_list))
            total_auc = metrics.auc(fpr, tpr)

            """
            compute thereshold, and then compute the accuracy of AMD and Myopia
            """
            percentage = 0.75
            _threshold_for_acc = sorted(normal_train_pred_list)[int(len(normal_train_pred_list) * percentage)]

            normal_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in normal_pred_list]
            amd_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in amd_pred_list]
            myopia_cls_pred_list = [(0 if i < _threshold_for_acc else 1) for i in myopia_pred_list]

            # acc, sensitivity and specifity
            def calcu_cls_acc(pred_list, gt_list):
                cls_pred_list = normal_cls_pred_list + pred_list
                gt_list = normal_gt_list + gt_list
                acc = metrics.accuracy_score(y_true=gt_list, y_pred=cls_pred_list)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true=gt_list, y_pred=cls_pred_list).ravel()
                sen = tp / (tp + fn + 1e-7)
                spe = tn / (tn + fp + 1e-7)
                return acc, sen, spe

            amd_acc, amd_sen, amd_spe = calcu_cls_acc(amd_cls_pred_list, amd_gt_list)
            myopia_acc, myopia_sen, myopia_spe = calcu_cls_acc(myopia_cls_pred_list, myopia_gt_list)


            """
            plot metrics curve
            """
            # ROC curve
            self.vis.draw_roc(fpr, tpr)

            metrics_auc_str = 'AUC = {:.4f}, AMD AUC = {:.4f}, Myopia AUC = {:.4f}'.\
                format(total_auc, amd_auc, myopia_auc)
            metrics_amd_acc_str = '\n amd_acc = {:.4f}, amd_sen = {:.4f}, amd_spe = {:.4f}'.\
                format(amd_acc, amd_sen, amd_spe)
            metrics_myopia_acc_str = '\n myopia_acc = {:.4f}, myopia_sen = {:.4f}, myopia_spe = {:.4f}'.\
                format(myopia_acc,  myopia_sen, myopia_spe)

            self.vis.text(metrics_auc_str + metrics_amd_acc_str + metrics_myopia_acc_str)
            print(metrics_auc_str + metrics_amd_acc_str + metrics_myopia_acc_str)

    def forward_cls_dataloader(self, loader, is_disease):
        gt_list = []
        pred_list = []
        for i, (image, image_name_item) in enumerate(loader):
            image = image.cuda(non_blocking=True)
            # val, forward
            seg_mask, image_rec = self.model(image)

            if self.args.data_modality == 'fundus':
                case_name = ['']
                image_name = image_name_item
            else:
                case_name, image_name = image_name_item

            """
            preditction
            """
            # BCWH -> B, anomaly score
            image_residual = torch.abs(image_rec - image)
            image_diff_mae = image_residual.mean(dim=3).mean(dim=2).mean(dim=1)

            # image: tensor
            # image_name: list
            # image_name.shape[0]: batch
            gt_list += [1 if is_disease else 0] * len(image_name)
            pred_list += image_diff_mae.tolist()

            """
            visdom
            """
            if i % self.args.vis_freq_inval == 0:
                image = image[:self.args.vis_batch]
                image_rec = image_rec[:self.args.vis_batch].clamp(0, 1)
                image_diff = torch.abs(image - image_rec)

                """
                Difference: seg_mask is different between fundus and oct images
                """
                if self.args.data_modality == 'fundus':
                    seg_mask = seg_mask[:self.args.vis_batch].clamp(0, 1)
                else:
                    seg_mask = torch.argmax(seg_mask[:self.args.vis_batch], dim=1).float()
                    seg_mask = (seg_mask.unsqueeze(dim=1) / 11).clamp(0, 1)

                vim_images = torch.cat([image, seg_mask, image_rec, image_diff], dim=0)

                self.vis.images(vim_images, win_name='val', nrow=self.args.vis_batch)

                """
                save images
                """
                output_save = os.path.join(self.args.output_root,
                                           self.args.project,
                                           'output_v1_0812',
                                           '{}'.format(self.args.version),
                                           'val')

                if not os.path.exists(output_save):
                    os.makedirs(output_save)
                tv.utils.save_image(vim_images, os.path.join(
                    output_save, '{}_{}.png'.format(case_name[0], image_name[0])), nrow=self.args.vis_batch)

        return gt_list, pred_list


class MultiTestForFigures(object):
    def __init__(self):
        args = ParserArgs().args
        cuda_visible(args.gpu)

        cudnn.benchmark = True

        if args.data_modality == 'fundus':
            # IDRiD dataset for segmentation
            # image, mask, image_name_item

            # iSee dataset for classification
            # image, image_name
            self.train_loader, self.normal_test_loader, \
            self.amd_fundus_loader, self.myopia_fundus_loader = \
                ClassificationFundusDataloader(data_root=args.isee_fundus_root,
                                               batch=args.batch,
                                               scale=args.scale).data_load()

        else:
            # Challenge OCT dataset for classification
            # image, [case_name, image_name]
            self.train_loader, self.normal_test_loader, self.oct_abnormal_loader = OCT_ClsDataloader(
                                                    data_root=args.challenge_oct,
                                                       batch=args.batch,
                                                       scale=args.scale).data_load()

        print_args(args)
        self.args = args

        for ablation_mode in range(6):
            args.resume = 'v22_ablation_{}@fundus@woVGG/latest_ckpt.pth.tar'.format(ablation_mode)
            self.model = PNetModel(args)
            self.test_cls(ablation_mode)


    def test_cls(self, ablation_mode, original_flag=False):
        # self.model.eval()
        self.model.train()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            if self.args.data_modality == 'fundus':
                self.forward_cls_dataloader(
                    loader=self.myopia_fundus_loader,
                    ablation_mode=ablation_mode,
                    original_flag=original_flag)

                self.forward_cls_dataloader(
                    loader=self.amd_fundus_loader,
                    ablation_mode=ablation_mode,
                    original_flag=original_flag)
            else:
                raise NotImplementedError('error')

            self.forward_cls_dataloader(
                loader=self.normal_test_loader,
                ablation_mode=ablation_mode,
                original_flag=original_flag)

    def forward_cls_dataloader(self, loader, ablation_mode, original_flag):
        for i, (image, image_name_item) in enumerate(loader):
            image = image.cuda(non_blocking=True)
            # val, forward
            seg_mask, image_rec = self.model(image)

            image_name = image_name_item

            """
            save images
            """
            output_save = os.path.join('/home/imed/new_disk/workspace/',
                                       self.args.project,
                                       'output_v1_0812',
                                       'ablation_study_fundus_ML_feature')

            if not os.path.exists(output_save):
                os.makedirs(output_save)
            if original_flag:
                tv.utils.save_image(image, os.path.join(
                    output_save, '{}_a_input.png'.format(image_name[0])))
                tv.utils.save_image(seg_mask, os.path.join(
                    output_save, '{}_b_mask.png'.format(image_name[0])))
            order = ['c', 'd', 'e', 'f', 'g', 'h', 'i'][ablation_mode]
            tv.utils.save_image(image_rec, os.path.join(
                output_save, '{}_{}_{}.png'.format(image_name[0], order, ablation_mode)))


if __name__ == '__main__':
    import pdb
    RunMyModel()
    # MultiTestForFigures()
