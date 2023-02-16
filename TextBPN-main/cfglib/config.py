from easydict import EasyDict
import torch
import os

config = EasyDict()

# LeeHakho
config.CRAFT = True

config.gpu = "0,1,2,3,4,5,6"

# max point per polygon for annotation
config.max_points = 20 #4 -> 20

# control points number
config.num_points = 20

# dataloader jobs number
config.num_workers = 24

# batch_size
config.batch_size = 12

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 0.1

# using GPU
config.cuda = True

config.output_dir = 'output'

config.input_size = 640 #640 -> 224

# max polygon per image
# synText, total-text:600; CTW1500: 1200; icdar: ; MLT: ; TD500: .
config.max_annotation = 8

# adj num for graph
config.adj_num = 4

# use hard examples (annotated as '#')
config.use_hard = True

# prediction on 1/scale feature map
config.scale = 1

# # clip gradient of loss
config.grad_clip = 0

# demo tcl threshold
config.dis_threshold = 0.3

config.cls_threshold = 0.8

# Contour approximation factor
config.approx_factor = 0.007


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')