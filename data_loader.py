#!/usr/bin/env python
# -*-coding: utf -8-*-
"""
@ Author: ZhanYang
@ File Name: data_loader.py
@ Email: zhanyang@whut.edu.cn
@ Time: 2021/11/5 17:01
"""
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
import utils
from utils.transforms import letterbox

import matplotlib.pyplot as plt
import torch.utils.data as data
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, split, imsize=640, transform= None, augment= False,
                  testmode=False,max_query_len=40,  bert_model='bert-base-uncased'):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        self.query_len = max_query_len  
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        file = open('RSVGD\\' + split + '.txt',"r").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist(self.anno_path, '.xml')
        for annotation in annotations:
            root = ET.parse(annotation).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = str(images_path) + '/' + root.find("./filename").text
                    box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text),int(member[2][3].text)], dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1


    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)  # box format: to x1 y1 x2 y2
        img = cv2.imread(img_path)
        return img, phrase, bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        phrase_out = phrase
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True, True, True

        # seems a bug in torch transformation resize, so separate in advance
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros_like(img)

        img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
        bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
        bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
        # Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)

        # encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(examples=examples, seq_length=self.query_len,tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        word_split = features[0].tokens[1:-1]

        if self.testmode:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], phrase_out
        else:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

# Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


