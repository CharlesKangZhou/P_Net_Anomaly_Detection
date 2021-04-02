import os
from scipy import misc
from PIL import Image, ImageEnhance
import random
import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T


class SourceOCTloader(object):
    def __init__(self, data_root, batch, scale, workers=8, flip=None, rotate=0, enhance_p=0):
        self.data_root = data_root
        self.batch = batch
        self.workers = workers
        self.config = dict(
            scale=scale,
            flip=flip,
            rotate=rotate,
            enhance_p=enhance_p
        )

    def data_load(self):
        train_set = SourceOCTdataset(
            data_root=self.data_root,
            mode='train',
            config=self.config
        )

        test_set = SourceOCTdataset(
            data_root=self.data_root,
            mode='test',
            config=self.config
        )

        all_set = data.ConcatDataset([train_set, test_set])

        train_loader = data.DataLoader(
            dataset=all_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True,
            shuffle=True
        )

        return train_loader


class ChallengeOCTloader(object):
    def __init__(self, data_root, batch, scale, workers=8, flip=None, rotate=0, enhance_p=0, stage=1):
        """
        The dataloader for dataset which with lesion but without structure
        :param data_root: the root path of the dataset
        :param batch:
        :param scale: image resolution
        :param workers: (int, optional) how many subprocesses to use for data
            loading. (default: 8)
        :param flip: (bool) data augmentation for train, left-right flip. (default: False)
        :param rotate:
        :param enhance_p:
        """
        self.data_root = data_root
        self.batch = batch

        self.workers = workers
        self.config = dict(
            scale=scale,
            flip=flip,
            rotate=rotate,
            enhance_p=enhance_p
        )

        self.stage = stage

    def data_load(self):
        train_set = ChanllengeOCTdataset(
            data_root=self.data_root,
            scale=self.config['scale'],
            mode='train',
            flip=self.config['flip'],
            rotate=self.config['rotate'],
            enhance_p=self.config['enhance_p'],
            stage=self.stage
        )

        test_set = ChanllengeOCTdataset(
            data_root=self.data_root,
            scale=self.config['scale'],
            mode='test'
        )

        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True,
            shuffle=True
        )

        test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True,
            # shuffle=True
        )

        return train_loader, test_loader


class SourceOCTdataset(data.Dataset):
    def __init__(self, data_root, mode, config):
        super(SourceOCTdataset, self).__init__()
        self.data_root = data_root
        assert mode in ['train', 'test'], 'error in mode, got {}'.format(mode)
        self.mode = mode
        scale = config['scale']
        self.flip = config['flip']
        self.rotate = config['rotate']
        self.enhance_p = config['enhance_p']

        case_name_list = os.listdir(os.path.join(data_root, mode))
        self.case_image_name_list = []
        for case_name_item in case_name_list:
            image_name_list = os.listdir(os.path.join(data_root, mode, case_name_item, 'mask_11'))
            self.case_image_name_list += [[case_name_item, item] for item in image_name_list]

        # image and mask transform
        self.image_t = T.Compose([
            T.Resize((scale, scale)),
            T.ToTensor()
        ])
        self.mask_t = T.Compose([
            T.Resize((scale, scale)),
        ])

    def __getitem__(self, item):
        case_name, image_name = self.case_image_name_list[item]
        mask_path = os.path.join(self.data_root, self.mode, case_name, 'mask_11', image_name)
        image_path = os.path.join(self.data_root, self.mode, case_name, image_name)

        mask = Image.open(mask_path).convert('L')
        image = Image.open(image_path)

        # data augumentation for Anomaly Detection no matter train or test
        if self.mode == 'train' or self.mode == 'test':
            if random.random() < self.enhance_p:
                image = RandomEnhance(image)
            if self.flip and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rotate > 0:
                angle = random.randint(-self.rotate, self.rotate)
                image = image.rotate(angle)
                mask = mask.rotate(angle)

        if self.image_t is not None:
            image = self.image_t(image)

        if self.mask_t is not None:
            mask = self.mask_t(mask)
            mask = torch.Tensor(np.array(mask))

        return image, mask, [case_name, image_name]

    def __len__(self):
        return len(self.case_image_name_list)


class ChanllengeOCTdataset(data.Dataset):
    def __init__(self, data_root, scale, mode, flip=False, rotate=0, enhance_p=0, stage=1):
        super(ChanllengeOCTdataset, self).__init__()

        # (default) Stage 1, all train set is source domain, mask_dir = 'images'
        # Stage 2, all crf train set

        self.data_root = data_root
        assert mode in ['train', 'test'], 'error in mode, got {}'.format(mode)
        self.mode = mode
        self.flip = flip
        self.rotate = rotate
        self.enhance_p = enhance_p

        # to optimize
        assert stage in [1, 2, 3], 'error in stage'
        if mode == 'train' and stage == 1:  # source domain
            self.case_dir = 'images'
            self.mask_dir = 'images'
        elif mode == 'train':  # CRF retrain
            self.case_dir = 'layer_mask_crf'
            self.mask_dir = 'layer_mask_crf'
        else:  # test
            self.case_dir = 'images'
            self.mask_dir = 'lesion_mask'

        case_name_list = os.listdir(os.path.join(data_root, mode, self.case_dir))
        self.case_image_name_list = []
        for case_name_item in case_name_list:
            image_name_list = os.listdir(os.path.join(data_root, mode, self.case_dir, case_name_item))
            self.case_image_name_list += [(case_name_item, item) for item in image_name_list]

        # image and mask transform
        self.image_t = T.Compose([
            T.Resize((scale, scale)),
            T.ToTensor()
        ])
        self.mask_t = T.Compose([
            T.Resize((scale, scale)),
        ])

    def __getitem__(self, item):
        case_name, image_name = self.case_image_name_list[item]
        mask_path = os.path.join(self.data_root, self.mode, self.mask_dir, case_name, image_name)
        image_path = os.path.join(self.data_root, self.mode, 'images', case_name, image_name)

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path)

        # data augumentation for Anomaly Detection no matter train or test
        if self.mode == 'train':
            if random.random() < self.enhance_p:
                image = RandomEnhance(image)
            if self.flip and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rotate > 0:
                angle = random.randint(-self.rotate, self.rotate)
                image = image.rotate(angle)
                mask = mask.rotate(angle)

        if self.image_t is not None:
            image = self.image_t(image)

        if self.mask_t is not None:
            mask = self.mask_t(mask)
            mask = torch.Tensor(np.array(mask))

        return image, mask, [case_name, image_name]

    def __len__(self):
        return len(self.case_image_name_list)


class OCT_ClsTrainSet(data.Dataset):
    def __init__(self, data_root, scale, mode='train', flip=False, rotate=0, enhance_p=0):
        super(OCT_ClsTrainSet, self).__init__()

        self.data_root = data_root
        self.flip = flip
        self.rotate = rotate
        self.enhance_p = enhance_p

        self.case_dir = 'layer_mask_crf'

        case_name_list = os.listdir(os.path.join(data_root, mode, self.case_dir))
        self.case_image_name_list = []
        for case_name_item in case_name_list:
            image_name_list = os.listdir(os.path.join(data_root, mode, self.case_dir, case_name_item))
            self.case_image_name_list += [(case_name_item, item) for item in image_name_list]

        # image and mask transform
        self.image_t = T.Compose([
            T.Resize((scale, scale)),
            T.ToTensor()
        ])
        self.mask_t = T.Compose([
            T.Resize((scale, scale)),
        ])

    def __getitem__(self, item):
        case_name, image_name = self.case_image_name_list[item]
        image_path = os.path.join(self.data_root, 'train', 'images', case_name, image_name)

        image = Image.open(image_path).convert('L')

        ## _todo: data augumentation for Anomaly Detection no matter train or test
        # if self.mode == 'train':
        #     if random.random() < self.enhance_p:
        #         image = RandomEnhance(image)
        #     if self.flip and random.random() > 0.5:
        #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #     if self.rotate > 0:
        #         angle = random.randint(-self.rotate, self.rotate)
        #         image = image.rotate(angle)
        #         mask = mask.rotate(angle)

        if self.image_t is not None:
            image = self.image_t(image)

        return image, [case_name, image_name]

    def __len__(self):
        return len(self.case_image_name_list)


class OCT_ClsDataset(data.Dataset):
    def __init__(self, data_root, scale, is_disease):
        super(OCT_ClsDataset, self).__init__()
        cases_dir = 'images' if is_disease else 'normal_images'
        self.images_path_list = []
        for case_item in os.listdir(os.path.join(data_root, 'test', cases_dir)):
            self.images_path_list += \
                glob.glob('{}/*'.format(os.path.join(data_root, 'test', cases_dir, case_item)))

        # image transform
        self.image_t = T.Compose([
            T.Resize((scale, scale)),
            T.ToTensor()
        ])

    def __getitem__(self, item):
        image_path = self.images_path_list[item]
        case_name, image_name = image_path.split('/')[-2:]

        image = Image.open(image_path).convert('L')

        if self.image_t is not None:
            image = self.image_t(image)

        return image, [case_name, image_name]

    def __len__(self):
        return len(self.images_path_list)


class OCT_ClsDataloader(object):
    def __init__(self, data_root, batch, scale):
        """
        The dataloader for dataset which with lesion but without structure
        :param data_root: the root path of the dataset
        :param batch:
        :param scale: image resolution
        :param workers: (int, optional) how many subprocesses to use for data
            loading. (default: 8)
        :param flip: (bool) data augmentation for train, left-right flip. (default: False)
        :param rotate:
        :param enhance_p:
        """
        self.data_root = data_root
        self.batch = batch

        self.config = dict(
            scale=scale,
            # flip=flip,
            # rotate=rotate,
            # enhance_p=enhance_p
        )

    def data_load(self):
        train_set = OCT_ClsTrainSet(
            data_root=self.data_root,
            scale=self.config['scale'],
            mode='train',
            # flip=self.config['flip'],
            # rotate=self.config['rotate'],
            # enhance_p=self.config['enhance_p']
        )

        normal_test_set = OCT_ClsDataset(
            data_root=self.data_root,
            scale=self.config['scale'],
            is_disease=False
        )

        abnormal_set = OCT_ClsDataset(
            data_root=self.data_root,
            scale=self.config['scale'],
            is_disease=True
        )

        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            num_workers=8,
            pin_memory=True,
            shuffle=True
        )

        normal_test_loader = data.DataLoader(
            dataset=normal_test_set,
            batch_size=self.batch,
            num_workers=8,
            pin_memory=True,
        )

        abnormal_loader = data.DataLoader(
            dataset=abnormal_set,
            batch_size=self.batch,
            num_workers=8,
            pin_memory=True,
        )

        return train_loader, normal_test_loader, abnormal_loader


# from PIL import ImageEnhance
# import random

def RandomEnhance(image):
    factor = random.uniform(0.8, 1.8)
    random_seed = random.randint(1, 3)
    if random_seed == 1:
        img_enhanced = ImageEnhance.Brightness(image)
    elif random_seed == 2:
        img_enhanced = ImageEnhance.Contrast(image)
    else:
        img_enhanced = ImageEnhance.Sharpness(image)
    image = img_enhanced.enhance(factor)
    return image



