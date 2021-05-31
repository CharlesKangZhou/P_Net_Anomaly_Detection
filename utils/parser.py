# -*- coding: utf-8 -*-
import argparse
import warnings
import os


class ParserArgs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='PyTorch Training and Testing'
        )

        # self.parser addition
        self.constant_init()
        self.get_general_parser()
        self.get_data_parser()
        self.get_parms_parser()
        self.get_freq_and_other_parser()

        # ablation exps
        self.get_ablation_exp_args()
        # comparison exps
        self.get_comparison_exps_args()

    def constant_init(self):
        # path and server
        self.ai_data_root = '/p300/'
        self.ws_data_root = '/home/imed/new_disk/'
        self.ai_output_root = '/p300/outputspace/'
        self.ws_output_root = '/home/imed/new_disk/workspace/'

        # constant
        self.parser.add_argument('--project', default='P-Net',
                            help='project name in workspace')
        self.parser.add_argument('--d2g_lr', type=float, default=0.1,
                                 help='discriminator/generator learning rate ratio')
        self.parser.add_argument('--weight_decay', default=1e-4, type=float,
                                 metavar='W', help='weight decay (default: 1e-4)')

    def get_general_parser(self):
        # general useful args
        self.parser.add_argument('--version', default='v99_debug',
                                 help='the version of different method/setting/parameters etc')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (format: version/path.tar)')
        self.parser.add_argument('--predict', action='store_true',
                            help='predict mode, rather than train and val')
        self.parser.add_argument('--port', default=31430, type=int, help='visdom port')
        self.parser.add_argument('--gpu', nargs='+', type=int,
                            help='gpu id for single/multi gpu')

    def get_data_parser(self):
        # dataset
        self.parser.add_argument('--data_modality', choices=['oct', 'fundus'],
                            help='the modality of data. No default.')
        self.parser.add_argument('--scale', default=224, type=int,
                                 help='image scale (default: 224)')

        # # data augmentation
        self.parser.add_argument('--enhance_p', type=float, default=0)
        self.parser.add_argument('--flip', type=bool, default=False)
        self.parser.add_argument('--rotate', type=int, default=0)

    def get_parms_parser(self):
        # model hyper-parameters
        self.parser.add_argument('--start_epoch', default=0, type=int,
                            help='numbet of start epoch to run')
        self.parser.add_argument('--n_epochs', default=800, type=int, metavar='N',
                            help='number of total epochs to run')
        self.parser.add_argument('--batch', default=8, type=int,
                            metavar='N', help='mini-batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                            metavar='LR', help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')

        self.parser.add_argument('--b1', type=float, default=0.1, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--lamd_p', '--lamd_pixel', default=100, type=float,
                            help='Loss weight of L1 pixel-wise loss between translated image and real image')
        self.parser.add_argument('--lamd_fm', default=0, type=float)
        self.parser.add_argument('--lamd_gen', default=1, type=float)


    def get_ablation_exp_args(self):
        self.parser.add_argument('--DA_isee', '--DA_ablation_mode_isee', default=0.0001, type=float,
                                 choices=[0, 0.001, 0.0001],
                                 help='the g weight for domain adaptation')
        self.parser.add_argument('--ablation_mode', default=6, choices=[0, 2, 4],
                            type=int, help='ablation study for multi-level feature')
        self.parser.add_argument('--lamd_mask_fusion', default=0, type=float,)
        self.parser.add_argument('--stage3_epoch', default=20, type=int)
        self.parser.add_argument('--lamd_mask', default=10, type=float,
                                 help='range = (0, 1)')

    def get_comparison_exps_args(self):
        self.parser.add_argument('--com_model_name', choices=['ae', 'ae_gan', 'pix_gan', 'ganomaly',
                                                        'cycle_gan'], type=str,
                                 help='the comparison exps name')
        self.parser.add_argument('--has_D', default=False, action='store_true',)

    # using this function when define the obeject of the class
    def get_args(self):
        import socket
        args = self.parser.parse_args()
        args.node = socket.gethostname()

        ws_name = args.node.split('-')[0]
        # zhoukang_XX,  zhoukang-XX---------------------------------------------------------------------------------+
        if ws_name == 'zhoukang' or len(ws_name) > 8:
            args.vis_server = 'http://10.10.10.100'
            args.challenge_oct = os.path.join(self.ai_data_root, 'dataset_eye/Eye_Public_Dataset/ai_oct_challenge')
        else:
            # imed-007
            args.server = 'ws'
            args.cheng_oct = os.path.join(self.ai_data_root, 'imed_dataset/Topcon_Normal_AROD/signal_crop_512')
            args.challenge_oct = os.path.join(self.ai_data_root, 'eye_dataset/Eye_Public_Dataset/ai_oct_challenge')
            args.fundus_data_root = os.path.join(self.ws_data_root,
                                                 'eye_dataset/Eye_Public_Dataset/AnomalyFundusLesion')
            args.isee_fundus_root = os.path.join(self.ws_data_root, 'imed_dataset/iSee_anomaly/preprocess')
            args.output_root = self.ws_output_root
            args.vis_server = 'http://localhost'

        self.assert_version(args.version)

        return args

    def get_freq_and_other_parser(self):
        # other useful args
        self.parser.add_argument('--validate_freq', default=10, type=int,
                                 help='validate frequency (default: 5)')
        self.parser.add_argument('--validate_start_epoch', default=30, type=int)
        self.parser.add_argument('--validate_each_epoch', default=30, type=int)
        self.parser.add_argument('--vis_freq', default=10, type=int,
                                help='data sent frequency to visdom server')
        self.parser.add_argument('--vis_batch', default=4, type=int)
        self.parser.add_argument('--vis_freq_inval', default=5, type=int)
        self.parser.add_argument('--save_freq', default=60, type=int,
                            help='validate frequency (default: n_epochs / 5 and times of val_freq)')
        self.parser.add_argument('--print_freq', default=90, type=int,
                            metavar='N', help='print frequency (default: 90)')

    @staticmethod
    def assert_version(version):
        # format: v01_XXX&XXX&sss_XXX
        v_split_list = version.split('_')
        v_major = v_split_list[0][0] == 'v' and v_split_list[0][1:].isdigit() and len(v_split_list[0]) == 3
        if not v_major:
            warnings.warn('The version name is warning')
