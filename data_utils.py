'''
Description:  
Author: yinwen
Date: 2023-02-01 14:12:37
'''
import os
import sys
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, DebertaV2Tokenizer
from torch.utils.data import Dataset


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel
                short = list(d['short'])

                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list, 'short': short}
                all_data.append(sample)

    return all_data


def ParseRawData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel

                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class Tokenizer4RoBerta:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class Tokenizer4DeBerta:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = DebertaTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAPROData(Dataset):
    def __init__(self, fname, tokenizer, opt, train_pre=None):
        self.rawdata = []
        self.data = []
        parse = ParseRawData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            left_tokens, term_tokens, right_tokens = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)

            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 4:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                else:
                    right_tokens.pop()

            rawtext_tokens = left_tokens + term_tokens + right_tokens
            context_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(rawtext_tokens) + \
                          [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                          [tokenizer.mask_token_id] + [tokenizer.sep_token_id]

            label_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(rawtext_tokens) + \
                          [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                          [tokenizer.convert_tokens_to_ids(obj['label'])] + [tokenizer.sep_token_id]

            context_len = len(context_ids)
            mask_index = context_len - 2

            paddings = [0] * (tokenizer.max_seq_len - context_len)
            rawtext_len = len(rawtext_tokens)
            context_second_ids = [0] * (1 + rawtext_len + 1) + [1] * (len(term_tokens) + 2) + paddings
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_attention_mask = [1] * context_len + paddings # allcontext_mask
            context_ids += paddings
            label_ids += paddings
            context_ids = np.asarray(context_ids, dtype='int64')
            context_second_ids = np.asarray(context_second_ids, dtype='int64')
            context_attention_mask = np.asarray(context_attention_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            label_ids = np.asarray(label_ids, dtype='int64')

            data = {
                'input_ids': context_ids,
                'token_type_ids': context_second_ids,
                'attention_mask': context_attention_mask,
                'label_ids': label_ids,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'mask_index': mask_index,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.rawdata.append(data)
        if train_pre is not None:
            train_size = int(train_pre * len(self.rawdata))
            self.data = self.rawdata[:train_size]
        else:
            self.data = self.rawdata
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ABSARoBertaData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseRawData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            left_tokens, term_tokens, right_tokens = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)

            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 4:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                else:
                    right_tokens.pop()

            rawtext_tokens = left_tokens + term_tokens + right_tokens
            context_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(rawtext_tokens) + \
                          [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                          [tokenizer.mask_token_id] + [tokenizer.eos_token_id]

            label_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(rawtext_tokens) + \
                          [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                          [tokenizer.convert_tokens_to_ids(obj['label'])] + [tokenizer.eos_token_id]

            context_len = len(context_ids)
            mask_index = context_len - 2

            paddings = [0] * (tokenizer.max_seq_len - context_len)
            rawtext_len = len(rawtext_tokens)
            context_second_ids = [0] * (1 + rawtext_len + 1) + [1] * (len(term_tokens) + 2) + paddings
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_attention_mask = [1] * context_len + paddings # allcontext_mask
            context_ids += paddings
            label_ids += paddings
            context_ids = np.asarray(context_ids, dtype='int64')
            context_second_ids = np.asarray(context_second_ids, dtype='int64')
            context_attention_mask = np.asarray(context_attention_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            label_ids = np.asarray(label_ids, dtype='int64')

            data = {
                'input_ids': context_ids,
                'token_type_ids': context_second_ids,
                'attention_mask': context_attention_mask,
                'label_ids': label_ids,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'mask_index': mask_index,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

