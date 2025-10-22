import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MPDataset(Dataset):

    def __init__(self, path_vid, datamode='style+content'):

        with open('./dataset/dict_vid_audioconvfea.pkl', "rb") as fr:
            self.dict_vid_convfea = pickle.load(fr)

        self.data_complete = pd.read_json('./dataset/data_gpt.json', orient='records', dtype=False, encoding='utf-8')

        self.frame_fea_patch = './dataset/ptvgg19_frames/'
        self.c3d_fea_patch = './dataset/c3d/'

        self.vid = []

        # video_id list
        with open('./dataset/vids/' + path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.datamode = datamode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # pre_score
        if item['response']['pre_score'] == "" or item['response']['pre_score'] == "error":
            pre_score = 0
        else:
            pre_score = float(item['response']['pre_score'])


        pre_score = torch.tensor(pre_score)

        # gpt_label
        if item['response']['pre_label'] == 'error':
            gpt_label = 1
        else:
            gpt_label = 0 if item['response']['pre_label'] == 'real' else 1
        gpt_label = torch.tensor(gpt_label)

        # label
        label = 0 if item['real_label'] == '真' else 1
        label = torch.tensor(label)

        # text
        title_tokens = self.tokenizer(item['title'] + '  ' + item['ocr'], max_length=512,
                                      padding='max_length', truncation=True)
        if self.datamode == 'style+content':
            style_tokens = self.tokenizer(item['response']['style'], max_length=512,
                                          padding='max_length', truncation=True)
            content_tokens = self.tokenizer(item['response']['content'], max_length=512,
                                            padding='max_length', truncation=True)
            match_tokens = self.tokenizer(item['response']['match_describe'], max_length=512,
                                          padding='max_length', truncation=True)
        elif self.datamode == 'style':
            style_tokens = self.tokenizer(item['response']['style'], max_length=512,
                                          padding='max_length', truncation=True)
            content_tokens = self.tokenizer('', max_length=512,
                                            padding='max_length', truncation=True)
        elif self.datamode == 'content':
            style_tokens = self.tokenizer('', max_length=512,
                                          padding='max_length', truncation=True)
            content_tokens = self.tokenizer(item['response']['content'], max_length=512,
                                            padding='max_length', truncation=True)

        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])
        style_inputid = torch.LongTensor(style_tokens['input_ids'])
        style_mask = torch.LongTensor(style_tokens['attention_mask'])
        content_inputid = torch.LongTensor(content_tokens['input_ids'])
        content_mask = torch.LongTensor(content_tokens['attention_mask'])
        match_inputid = torch.LongTensor(match_tokens['input_ids'])
        match_mask = torch.LongTensor(match_tokens['attention_mask'])

        # audio
        audioframes = self.dict_vid_convfea[vid]
        audioframes = torch.FloatTensor(audioframes)

        # frames
        frames = pickle.load(open(os.path.join(self.frame_fea_patch, vid + '.pkl'), 'rb'))
        frames = torch.FloatTensor(frames)

        # video
        c3d = h5py.File(self.c3d_fea_patch + vid + ".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        return {
            'label': label,
            'gpt_label': gpt_label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'style_inputid': style_inputid,
            'style_mask': style_mask,
            'content_inputid': content_inputid,
            'content_mask': content_mask,
            'match_inputid': match_inputid,
            'match_mask': match_mask,
            'frames': frames,
            'c3d': c3d,
            'pre_score': pre_score
        }


