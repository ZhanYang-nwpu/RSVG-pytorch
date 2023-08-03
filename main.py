#!/usr/bin/env python
# -*-coding: utf -8-*-
"""
@ Author: ZhanYang
@ File Name: data_loader.py
@ Github: https://github.com/ZhanYang-nwpu/RSVG-pytorch
@ Paper: https://ieeexplore.ieee.org/document/10056343
@ Dataset: https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing
"""
import argparse

import numpy as np
import pickle
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import logging
import datetime
import matplotlib as mpl
mpl.use('Agg')
import sys
import os
import time
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.optim

from data_loader import *
from models.model import MGVLF
from models.loss import Reg_Loss, GIoU_Loss
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

def main():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--images_path', type=str, default='DIOR_RSVG\\JPEGImages',
                        help='path to dataset splits data folder')
    parser.add_argument('--anno_path', type=str, default='DIOR_RSVG\\Annotations',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--time', default=40, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=0, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=150, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_dec', default=0.1, type=float, help='decline of learning rate')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',  #
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='train')
    # parser.add_argument('--test', dest='test', default=True, action='store_true', help='test')
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true', help='if tunebert')

    # * DETR
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400 + 40 + 1, type=int,
                        help="Number of query slots in VLFusion")
    parser.add_argument('--pre_norm', action='store_true')

    global args, anchors_full
    args = parser.parse_args()

    print('----------------------------------------------------------------------')
    print('模型参数：', args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ## fix random seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    ## save logs
    if args.savename == 'default':
        args.savename = 'MGVLF_batch%d_epoch%d_lr%d_seed%d' % (args.batch_size, args.nb_epoch,args.lr, args.seed)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s" % args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    val_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'val',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    test_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'test',
                         testmode=True,
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    print('trainset:', len(train_dataset), 'validationset:', len(val_dataset), 'testset:', len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

if __name__ == "__main__":
    main()
