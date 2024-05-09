import os
from os.path import join, isfile, splitext, basename

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data
import torch.utils.data as data
from PIL import Image

from .polar_transformations import centroid, to_polar


def norm01(x):
    return np.clip(x, 0, 255) / 255


# cross validation
class ISICDataset(data.Dataset):
    def __init__(self, config, train, aug=False, polar=False):
        super(ISICDataset, self).__init__()
        self.polar = polar
        self.train = train
        support_types = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        support_types = set(support_types)
        # load images, label, point
        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        base_path = join(config["dataset_path"], "train" if train else "val")
        self.origin_image_path = join(base_path, "images")
        self.ground_truth_path = join(base_path, "masks")
        self.gt_format = config["gt_format"]
        self.image_paths = [join(self.origin_image_path, f) for f in os.listdir(self.origin_image_path)
                            if isfile(join(self.origin_image_path, f)) and splitext(f)[1][1:] in support_types]


        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug
        self.size = config["image_size"]
        self.gt_format = config["gt_format"]

        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            #             A.RandomBrightnessContrast(p=p),
        ])

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        try:
            return self.gt_format.format(origin_image_name)
        except:
            return origin_image_name + "_segmentation.png"

    def __getitem__(self, index):
        # print(self.image_paths[index])
        # image = cv2.imread(self.image_paths[index])
        # image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label_data = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)
        image_path = self.image_paths[index]
        origin_image_name, extension = splitext(basename(image_path))
        filename = self.get_gt_file_name(origin_image_name, extension)
        msk_path = join(self.ground_truth_path, filename)
        image_data = np.array(Image.open(image_path).convert('RGB'))
        label_data = (np.array(Image.open(msk_path).convert('L')))

        label_data = np.array(
            cv2.resize(label_data, (self.size, self.size), cv2.INTER_NEAREST))
        point_data = cv2.Canny(label_data, 0, 255) / 255.0 > 0.5
        label_data = label_data / 255. > 0.5
        image_data = np.array(
            cv2.resize(image_data, (self.size, self.size), cv2.INTER_LINEAR))
        # if self.train:
        #     filter_point_data = (np.load(self.point_paths[index]) >
        #                          0.7).astype('uint8')
        #     filter_point_data = np.array(
        #         cv2.resize(filter_point_data, (self.size, self.size),
        #                    cv2.INTER_NEAREST))
        # else:
        filter_point_data = point_data.copy()

        # image_data = np.load(self.image_paths[index])
        # label_data = np.load(self.label_paths[index]) > 0.5
        # point_data = np.load(self.point_paths[index]) > 0.5
        # point_All_data = np.load(self.point_All_paths[index]) > 0.5  #

        #         label_data = np.expand_dims(label_data,-1)
        #         point_data = np.expand_dims(point_data,-1)
        if self.aug and self.train:
            mask = np.concatenate([
                label_data[..., np.newaxis].astype('uint8'),
                point_data[..., np.newaxis], filter_point_data[..., np.newaxis]
            ],
                axis=-1)
            #             print(mask.shape)
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)
            image_data, mask_aug = tsf['image'], tsf['mask']
            label_data = mask_aug[:, :, 0]
            point_data = mask_aug[:, :, 1]
            filter_point_data = mask_aug[:, :, 2]

        image_data = norm01(image_data)

        if self.polar:
            center = centroid(image_data)
            image_data = to_polar(image_data, center)
            label_data = to_polar(label_data, center) > 0.5

        label_data = np.expand_dims(label_data, 0)
        point_data = np.expand_dims(point_data, 0)
        filter_point_data = np.expand_dims(filter_point_data, 0)  #

        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        point_data = torch.from_numpy(point_data).float()
        filter_point_data = torch.from_numpy(filter_point_data).float()  #

        image_data = image_data.permute(2, 0, 1)
        return {
            # 'image_path': self.image_paths[index],
            # 'label_path': self.label_paths[index],
            # 'point_path': self.point_paths[index],
            'image': image_data,
            'label': label_data,
            'point': point_data,
            'filter_point_data': filter_point_data,
            'origin_image_name': origin_image_name
        }

    def __len__(self):
        return self.num_samples
