"""
@ Author: ZhanYang
@ File Name: data_loader.py
@ Email: zhanyang@mail.nwpu.edu.cn
@ Github: https://github.com/ZhanYang-nwpu/RSVG-pytorch
@ Paper: https://ieeexplore.ieee.org/document/10056343
@ Dataset: https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from models.CNN_MGVLF import build_VLFusion,build_CNN_MGVLF

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def generate_coord(batch, height, width):
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model


class MGVLF(nn.Module):
    def __init__(self, bert_model='bert-base-uncased',tunebert=True, args=None):
        super(MGVLF, self).__init__()
        self.tunebert = tunebert
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024

        # Visual model
        self.visumodel = build_CNN_MGVLF(args)
        self.visumodel = load_weights(self.visumodel, './saved_models/detr-r50-e632da11.pth')
        
        # Text model
        self.textmodel = BertModel.from_pretrained('bert-base-uncased')

        # Multimodal Fusion Module
        self.vlmodel = build_VLFusion(args)
        self.vlmodel = load_weights(self.vlmodel, './saved_models/detr-r50-e632da11.pth')
        
        # Localization Module
        self.Prediction_Head = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 4),)
        for p in self.Prediction_Head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, mask, word_id, word_mask):
        # Multimodal Encoder —— Language
        all_encoder_layers, pooled_output = self.textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
        sentence_frature = pooled_output

        fl = (all_encoder_layers[-1] + all_encoder_layers[-2]+ all_encoder_layers[-3] + all_encoder_layers[-4])/4
        if not self.tunebert:  # fix bert during training
            fl = fl.detach()

        # Multimodal Encoder —— Multi-Granularity Visual Language Fusion Module
        fv = self.visumodel(image, mask, word_mask, fl, sentence_frature)

        # Multimodal Fusion Module
        x = self.vlmodel(fv, fl)

        # Localization Module
        outbox = self.Prediction_Head(x)  # (x; y;w; h)
        outbox = outbox.sigmoid()*2.-0.5  # (x; y; x; y)

        return outbox
