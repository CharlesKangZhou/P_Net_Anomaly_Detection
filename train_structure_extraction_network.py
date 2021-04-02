import os
import sys
import datetime
import time

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim

from dataloader.resc_dataloader import SourceOCTloader, ChallengeOCTloader
from networks.unet import UNet_4mp
from networks.discriminator import Discriminator
from utils.gan_loss import AdversarialLoss
from utils.visualizer import Visualizer
from utils.utils import adjust_lr, cuda_visible, print_args, save_ckpt
from utils.parser import ParserArgs
# from utils.crf import dense_crf


class SegTransferModel(nn.Module):
    def __init__(self, args):
        super(SegTransferModel, self).__init__()
        # n_classes for Fundus: 1
        # n_classes for OCT: 12
        self.args = args
        assert args.data_modality in ['oct', 'fundus'], 'error in seg_mode, got {}'.format(args.data_modality)

        # model on gpu
        if self.args.data_modality == 'fundus':
            model_G = UNet_4mp(n_channels=1, n_classes=1)
        else:
            model_G = UNet_4mp(n_channels=1, n_classes=12)
        model_D = Discriminator(in_channels=1)

        model_G = nn.DataParallel(model_G).cuda()
        model_D = nn.DataParallel(model_D).cuda()

        l1_loss = nn.L1Loss().cuda()
        nll_loss = nn.NLLLoss().cuda()
        adversarial_loss = AdversarialLoss().cuda()

        self.add_module('model_G', model_G)
        self.add_module('model_D', model_D)

        self.add_module('l1_loss', l1_loss)
        self.add_module('nll_loss', nll_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # optimizer
        self.optimizer_G = torch.optim.Adam(params=self.model_G.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr * args.d2g_lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        # Optionally resume from a checkpoint
        if self.args.resume:
            ckpt_root = os.path.join(args.output_root, args.project, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.model_G.load_state_dict(checkpoint['state_dict_G'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    def process(self, image_source, mask_source_gt, image_target):
        # ---------------
        #  Source Domain
        # ---------------

        # zero optimizers
        self.optimizer_G.zero_grad()

        # process_outputs
        output_source_mask = self(image_source)

        # generator l1 loss or nll_loss
        if self.args.data_modality == 'fundus':
            # output_source_mask: B1WH
            # mask_source_gt: B1WH
            seg_loss = self.l1_loss(output_source_mask, mask_source_gt) * self.args.lamd_p
        else:
            # output_source_mask: BCWH
            # mask_source_gt: BWH
            output_source_mask = F.log_softmax(output_source_mask, dim=1)
            seg_loss = self.nll_loss(output_source_mask, mask_source_gt.long()) * self.args.lamd_p

            # output_source_mask: BCWH -> BWH, float -> long
            # logits(probs) -> mask
            _, output_source_mask = torch.max(output_source_mask, dim=1)
            # BWH -> B1WH, long -> float
            output_source_mask = output_source_mask.float().unsqueeze(dim=1)

        # backward
        self.backward(gen_loss=seg_loss)

        # ---------------
        #  Target Domain
        # ---------------
        # zero optimizers
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # process_outputs
        output_target_mask = self(image_target)
        gen_loss = 0
        dis_loss = 0

        if self.args.data_modality == 'oct':
            # BCWH -> BWH
            _, output_target_mask = torch.max(output_target_mask, dim=1)
            # BWH -> B1WH
            output_target_mask = output_target_mask.float().unsqueeze(dim=1)

        # discriminator loss
        dis_input_real = output_source_mask.detach()
        dis_input_fake = output_target_mask.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = output_target_mask
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True) * self.args.lamd_gen
        gen_loss += gen_gan_loss

        # backward
        self.backward(gen_loss=gen_loss, dis_loss=dis_loss)

        # create logs
        logs = dict(
            seg_loss=seg_loss,
            gen_loss=gen_loss,
            gen_gan_loss=gen_gan_loss,
            dis_loss=dis_loss,
        )

        return output_source_mask, output_target_mask, logs

    def forward(self, image):
        output_mask = self.model_G(image)
        return output_mask

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
            self.source_loader = AnoDRIVE_Loader(data_root=args.fundus_data_root,
                                                 batch=args.batch,
                                                 scale=args.scale,
                                                 pre=True       # pre-process
                                                 ).data_load()
            # self.target_loader, _ = AnoIDRID_Loader(data_root=args.fundus_data_root,
            #                                      batch=args.batch,
            #                                      scale=args.scale,
            #                                     pre=True).data_load()
            self.target_loader = NewClsFundusDataloader(data_root=args.isee_fundus_root,
                                                 batch=args.batch,
                                                 scale=args.scale).load_for_seg()

        else:
            self.source_loader = SourceOCTloader(data_root=args.cheng_oct,
                                                batch=args.batch,
                                                scale=args.scale,
                                                flip=args.flip,
                                                rotate=args.rotate,
                                                enhance_p=args.enhance_p).data_load()
            self.target_loader, _ = ChallengeOCTloader(data_root=args.challenge_oct,
                                                batch=args.batch,
                                                scale=args.scale).data_load()

        print_args(args)
        self.args = args
        self.new_lr = self.args.lr
        self.model = SegTransferModel(args)

        if args.predict:
            self.validate_loader(self.target_loader)
        else:
            self.train_validate()

    def train_validate(self):
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            _ = adjust_lr(self.args.lr, self.model.optimizer_G, epoch, [40, 80, 160, 240])
            new_lr = adjust_lr(self.args.lr, self.model.optimizer_D, epoch, [40, 80, 160, 240])
            self.new_lr = min(new_lr, self.new_lr)

            self.epoch = epoch

            self.train()
            if epoch % self.args.validate_freq == 0 and epoch > self.args.save_freq:
                self.validate()
                # self.validate_loader(self.normal_test_loader)
                # self.validate_loader(self.amd_fundus_loader)
                # self.validate_loader(self.myopia_fundus_loader)

            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Node: {}'.format(self.args.node))
            print('GPU: {}'.format(self.args.gpu))
            print('Version: {}\n'.format(self.args.version))

    def train(self):
        self.model.train()

        prev_time = time.time()

        target_loader_iter = self.target_loader.__iter__()
        # target_loader_isee_iter = self.target_loader.__iter__()
        for i, (image_source, mask_source_gt, _) in enumerate(self.source_loader):
            mask_source_gt = mask_source_gt.cuda(non_blocking=True)
            image_source = image_source.cuda(non_blocking=True).float()

            image_target, _ = next(target_loader_iter)
            image_target = image_target.cuda(non_blocking=True)
            output_source_mask, output_target_mask, logs = \
                self.model.process(image_source, mask_source_gt, image_target)

            # if self.epoch % 2 == 0:
            #     # train on IDRiD dataset
            #     image_target, _, _ = next(target_loader_iter)
            #     image_target = image_target.cuda(non_blocking=True)
            #     output_source_mask, output_target_mask, logs = \
            #         self.model.process(image_source, mask_source_gt, image_target)
            # else:
            #     # train on iSee dataset
            #     image_target, _, = next(target_loader_isee_iter)
            #     image_target = image_target.cuda(non_blocking=True)
            #     output_source_mask, output_target_mask, logs = \
            #         self.model.process(image_source, mask_source_gt, image_target)

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = self.epoch * self.source_loader.__len__() + i
            batches_left = self.args.n_epochs * self.source_loader.__len__() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                             (self.epoch, self.args.n_epochs,
                              i, self.source_loader.__len__(),
                              logs['dis_loss'].item(),
                              logs['gen_loss'].item(),
                              time_left))

            # --------------
            #  Visdom
            # --------------
            if i % self.args.vis_freq == 0:
                image_source = image_source[:self.args.vis_batch]
                image_target = image_target[:self.args.vis_batch]
                if self.args.data_modality == 'oct':
                    # OCT: {0, 1, ..., 11}, BWH
                    # BWH -> B1WH,
                    mask_source_gt = mask_source_gt[:self.args.vis_batch].unsqueeze(dim=1) / 11
                    # B1WH
                    output_source_mask = torch.clamp(output_source_mask[:self.args.vis_batch] / 11, 0, 1)
                    output_target_mask = torch.clamp(output_target_mask[:self.args.vis_batch] / 11, 0, 1)
                else:
                    # fundus: {0, 1}, B1WH
                    mask_source_gt = mask_source_gt[:self.args.vis_batch]
                    output_source_mask = torch.clamp(output_source_mask[:self.args.vis_batch], 0, 1)
                    output_target_mask = torch.clamp(output_target_mask[:self.args.vis_batch], 0, 1)

                vim_images = torch.cat([image_source,
                                        mask_source_gt,
                                        output_source_mask,
                                        image_target,
                                        output_target_mask], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

            if i+1 == self.source_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=logs['dis_loss'].item(),
                                             seg_loss=logs['seg_loss'].item(),
                                             lr=self.new_lr))
                self.vis.plot_single_win(dict(gen_loss=logs['gen_loss'].item(),
                                              gen_fm_loss=logs['gen_fm_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item(),
                                              gen_content_loss=logs['gen_content_loss'].item(),
                                              gen_style_loss=logs['gen_style_loss'].item()),
                                              win='gen_loss')

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for i, (image, _) in enumerate(self.target_loader):
                image = image.cuda(non_blocking=True).float()

                # forward
                output_mask = self.model(image)

                if i % self.args.vis_freq_inval == 0:
                    image = image[:self.args.vis_batch]
                    if self.args.data_modality == 'oct':
                        # OCT: {0, 1, ..., 11}
                        # gt: BWH
                        # model output: BCWH (C=12)

                        # BCWH -> BWH -> B1WH
                        output_mask = F.log_softmax(output_mask, dim=1)
                        _, output_mask = torch.max(output_mask, dim=1)
                        output_mask = output_mask.float().unsqueeze(dim=1)

                        # {0, 1, ..., 11} -> (0, 1)
                        output_mask = torch.clamp(output_mask[:self.args.vis_batch] / 11, 0, 1)
                    else:
                        # fundus: {0, 1}, B1WH
                        output_mask = output_mask[:self.args.vis_batch]

                    save_images = torch.cat([image, output_mask], dim=0)
                    output_save = os.path.join(self.args.output_root,
                                               self.args.project,
                                               'output',
                                               self.args.version,
                                               'val')
                    if not os.path.exists(output_save):
                        os.makedirs(output_save)
                    tv.utils.save_image(save_images, os.path.join(output_save, '{}.png'.format(i)),
                                        nrow=self.args.vis_batch)

                    # print('val: [Batch {}/{}]'.format(i, self.target_loader.__len__()))

        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_G': self.model.model_G.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  args=self.args)
        print('Save ckpt successfully!')

    def validate_loader(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for i, (image, image_name) in enumerate(dataloader):
                image = image.cuda(non_blocking=True).float()

                # forward
                output_mask = self.model(image)

                if i % self.args.vis_freq_inval == 0:
                    image = image[:self.args.vis_batch]
                    if self.args.data_modality == 'oct':
                        # OCT: {0, 1, ..., 11}
                        # gt: BWH
                        # model output: BCWH (C=12)

                        # BCWH -> BWH -> B1WH
                        output_mask = F.log_softmax(output_mask, dim=1)
                        _, output_mask = torch.max(output_mask, dim=1)
                        output_mask = output_mask.float().unsqueeze(dim=1)

                        # {0, 1, ..., 11} -> (0, 1)
                        output_mask = torch.clamp(output_mask[:self.args.vis_batch] / 11, 0, 1)
                    else:
                        # fundus: {0, 1}, B1WH
                        output_mask = output_mask[:self.args.vis_batch]

                    save_images = torch.cat([image, output_mask], dim=0)
                    output_save = os.path.join(self.args.output_root,
                                               self.args.project,
                                               'output',
                                               self.args.version,
                                               'val')
                    if not os.path.exists(output_save):
                        os.makedirs(output_save)
                    tv.utils.save_image(save_images, os.path.join(
                        output_save, '{}.png'.format(image_name[0])),
                                        nrow=self.args.vis_batch)


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for i, (image, _, item_name) in enumerate(self.target_loader):
                image = image.cuda(non_blocking=True).float()

                if self.args.batch == 1:
                    if self.args.data_modality == 'oct':
                        case_name, image_name = item_name
                        case_name = case_name[0]
                        image_name = image_name[0]
                    else:
                        case_name = 'fundus'
                        image_name = item_name[0]
                else:
                    raise NotImplementedError('error')

                # forward
                output_mask = self.model(image)

                dim_channel = 1
                if self.args.data_modality == 'oct':
                    # mask prob for CRF
                    mask_prob = F.softmax(output_mask, dim=dim_channel)

                    # output the segmentation mask
                    output_mask = F.log_softmax(output_mask, dim=dim_channel)
                    _, output_mask = torch.max(output_mask, dim=dim_channel)
                    output_mask = output_mask.float().unsqueeze(dim=dim_channel)
                    # {0, 1, ..., 11} -> (0, 1)
                    _output_mask = torch.clamp(output_mask / 11, 0, 1)

                    if self.args.use_crf:
                        # CHW -> HWC (224, 224, 1)
                        # optimize: tensor.permute(2, 0, 1)
                        _image = image.squeeze(dim=0).cpu().transpose(0, 2).transpose(0, 1)
                        # OCT, 1 channel. (224, 224, 1) -> (224, 224, 3)
                        _image = _image.repeat(1, 1, 3)
                        mask = mask_prob.squeeze(dim=0).cpu()
                        crf_mask = dense_crf(np.array(_image).astype(np.uint8), mask)
                        _crf_mask = torch.Tensor(crf_mask.astype(np.float)) / 11
                        # HW -> BCHW
                        _crf_mask = _crf_mask.expand((1, 1, -1, -1)).cuda()
                    else:
                        _crf_mask = output_mask

                else:
                    # fundus: {0, 1}, B1WH
                    _output_mask = output_mask.clamp(0, 1)
                    # raise NotImplementedError('error for fundus mode')

                save_images = torch.cat([image, _output_mask], dim=0)
                output_save_path = os.path.join('/home/imed/new_disk/workspace/',
                                           self.args.project,
                                           'output',
                                           self.args.version,
                                           'predict')
                save_name = '{}_{}.png'.format(case_name, image_name)
                self.vis.images(save_images, win_name='predict')
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
                tv.utils.save_image(save_images, os.path.join(output_save_path, save_name), nrow=2)

                pdb.set_trace()

                # ---------
                # save mask
                # ---------
                # To optimize
                save_flag = False
                if save_flag:
                    save_path = os.path.join(mask_vgg_root, case_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    self.save_oct(output_mask, os.path.join(save_path, image_name))

                    save_path = os.path.join(mask_crf_root, case_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    self.save_oct(crf_mask, os.path.join(save_path, image_name), crf_mode=True)


    def save_oct(self, tensor, filename, crf_mode=False):
        if crf_mode:
            misc.imsave(filename, tensor)
        else:
            B, C, _, _ = tensor.shape
            assert B == 1 and C ==1, 'error about shape'
            tensor = tensor.squeeze()
            ndarr = tensor.cpu().numpy()
            misc.imsave(filename, ndarr)


if __name__ == '__main__':
    import pdb
    from scipy import misc
    import numpy as np

    RunMyModel()

