# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2021/12/15 12:14
# @Software: PyCharm
# @Brief: 配置文件

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.01)

parser.add_argument('--dataset_train_dir', type=str,
                    default="/home/rhdai/workspace/code/image_classification/dataset/train/",
                    help='The directory containing the train data.')
parser.add_argument('--dataset_val_dir', type=str,
                    default="/home/rhdai/workspace/code/image_classification/dataset/validation/",
                    help='The directory containing the val data.')
parser.add_argument('--summary_dir', type=str, default="./summary/vit_base_patch16_224",
                    help='The directory of saving weights and tensorboard.')

# 预训练权重路径，如果不想载入就设置为空字符
parser.add_argument('--weights', type=str, default='./pretrain_weights/vit_base_patch16_224_in21k.pth',
                    help='Initial weights path.')

# 是否冻结权重
parser.add_argument('--freeze_layers', type=bool, default=True)
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='Select gpu device.')

parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                    help='The name of ViT model, Select one to train.')
parser.add_argument('--label_name', type=list, default=[
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
], help='The name of class.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
