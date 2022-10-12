# -*- coding: utf-8 -*-
# @File : dataloader.py
# @Author: Runist
# @Time : 2021/10/28 10:26
# @Software: PyCharm
# @Brief:
from torchvision import transforms, datasets
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, image_label, aug=False):
        self.image_label = image_label
        self.aug = aug

    def __getitem__(self, index):
        image, label = self.image_label[index]

        if self.aug:
            image = data_transform["train"](image)
        else:
            image = data_transform["val"](image)

        return image, label

    def __len__(self):
        return len(self.image_label)


class PathLoader(torch.utils.data.Dataset):

    def __init__(self, image_label_path, aug=False):
        self.image_label_path = image_label_path
        self.aug = aug

    def __getitem__(self, index):
        image_path, label = self.image_label_path[index]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.aug:
            image = data_transform["train"](image)
        else:
            image = data_transform["val"](image)

        return image, label

    def __len__(self):
        return len(self.image_label_path)


def get_data_loader(data_dir, batch_size, num_workers, aug=False):
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=aug,
                                         num_workers=num_workers)

    return loader, dataset
