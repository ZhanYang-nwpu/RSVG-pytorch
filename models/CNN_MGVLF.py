# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MLCM model and criterion classes.
@ Author: ZhanYang
@ File Name: data_loader.py
@ Email: zhanyang@mail.nwpu.edu.cn
@ Github: https://github.com/ZhanYang-nwpu/RSVG-pytorch
@ Paper: https://ieeexplore.ieee.org/document/10056343
@ Dataset: https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing
"""

import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_vis_transformer, build_transformer,build_de
from models.position_encoding import build_position_encoding


class CNN_MGVLF(nn.Module):
    """ This is the MLCM module """
    def __init__(self, backbone, transformer, DE, position_encoding):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.DE = DE
        hidden_dim = transformer.d_model
        self.pos = position_encoding
        self.text_pos_embed = nn.Embedding(40 + 1, hidden_dim)

        self.conv6_1 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=1, stride=1)
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.l_proj = torch.nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(), )

    def get_mask(self, nextFeatureMap, beforeMask):
        x = nextFeatureMap
        m = beforeMask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        return mask

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        features, pos = self.backbone(samples)
        featureMap4, mask4 = features[3].decompose()
        bs, c, h, w = featureMap4.shape

        conv6_1 = self.conv6_1(featureMap4)
        conv6_2 = self.conv6_2(conv6_1)
        conv7_1 = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(conv7_1)
        conv8_1 = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(conv8_1)

        conv5 = self.input_proj(featureMap4)
        fv1 = conv5.view(bs, 256, -1)
        fv2 = conv6_2.view(bs, 256, -1)
        fv3 = conv7_2.view(bs, 256, -1)
        fv4 = conv8_2.view(bs, 256, -1)
        fv2_mask = self.get_mask(conv6_2, mask4)
        fv3_mask = self.get_mask(conv7_2, fv2_mask)
        fv4_mask = self.get_mask(conv8_2, fv3_mask)

        pos1 = pos[-1]
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)
        fvpos1 = pos1.view(bs, 256, -1)
        fvpos2 = pos2.view(bs, 256, -1)
        fvpos3 = pos3.view(bs, 256, -1)
        fvpos4 = pos4.view(bs, 256, -1)

        fv = torch.cat((fv1, fv2), dim=2)
        fv = torch.cat((fv, fv3), dim=2)
        fv = torch.cat((fv, fv4), dim=2)
        fv = fv.permute(2, 0, 1)
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)
        fl = self.l_proj(textFeature)
        fl = fl.permute(1, 0, 2)
        fvl = torch.cat((fv, fl), dim=0)

        word_mask = word_mask.to(torch.bool)
        word_mask = ~word_mask
        sentence_mask = torch.zeros((bs, 1)).to(word_mask.device).to(torch.bool)
        text_mask = torch.cat((word_mask, sentence_mask), dim=1)
        vis_mask = torch.cat((mask4.view(bs, -1), fv2_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv3_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv4_mask.view(bs, -1)), dim=1)
        fvl_mask = torch.cat((vis_mask, text_mask), dim=1)

        flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        fvpos = torch.cat((fvpos1, fvpos2), dim=2)
        fvpos = torch.cat((fvpos, fvpos3), dim=2)
        fvpos = torch.cat((fvpos, fvpos4), dim=2)
        fvpos = fvpos.permute(2, 0, 1)
        fvlpos = torch.cat((fvpos, flpos), dim=0)

        out_layers = self.DE(fv1.permute(2, 0, 1), fvl, fvl_mask, fvlpos,fvpos1.permute(2, 0, 1))
        fv1_encode = out_layers[-1].permute(1, 2, 0)

        refineFeature = fv1_encode.view(bs, 256, h, w)
        out = self.transformer(refineFeature, mask4, pos1)
        return out


class VLFusion(nn.Module):
    def __init__(self, transformer, pos):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: no use
            """
        super().__init__()
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model
        self.pr = nn.Embedding(1, hidden_dim)

        self.v_proj = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),)
        self.l_proj = torch.nn.Sequential(
          nn.Linear(768, 256),
          nn.ReLU(),)

    def forward(self, fv, fl):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        bs, c, h, w = fv.shape
        _, _, l = fl.shape

        pv = self.v_proj(fv.view(bs, c, -1).permute(0,2,1))
        pl = self.l_proj(fl)
        pv = pv.permute(0,2,1)
        pl = pl.permute(0,2,1)

        pr = self.pr.weight
        pr = pr.expand(bs,-1).unsqueeze(2)

        x0 = torch.cat((pv, pl), dim=2)
        x0 = torch.cat((x0, pr), dim=2)
        
        pos = self.pos(x0).to(x0.dtype)
        mask = torch.zeros([bs, x0.shape[2]]).cuda()
        mask = mask.bool()
        out = self.transformer(x0, mask, pos)
        
        return out[-1]


def build_CNN_MGVLF(args):
    device = torch.device(args.device)
    backbone = build_backbone(args) # ResNet 50
    EN = build_vis_transformer(args)
    DE = build_de(args)
    pos = build_position_encoding(args, position_embedding='sine')

    model = CNN_MGVLF(
        backbone,
        EN,
        DE,
        pos,
    )
    return model


def build_VLFusion(args):
    device = torch.device(args.device)
    transformer = build_transformer(args)
    pos = build_position_encoding(args, position_embedding = 'learned')
    model = VLFusion(
        transformer,
        pos,
    )
    return model

