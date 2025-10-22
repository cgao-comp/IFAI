import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from .coattention import *
from .layers import *
from utils.metrics import *


class MainModel(torch.nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(MainModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.fuse_threshold = 0.8
        self.vote_threshold = 0.9
        self.text_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.mlp_out_dim = 384
        self.dim = fea_dim
        self.num_heads = 4

        self.dropout = dropout

        self.attention = Attention(dim=self.dim, heads=4, dropout=dropout)

        self.vggish_layer = torch.hub.load('./code/models/torchvggish/',
                                           'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                            pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(128, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_project = nn.Sequential(torch.nn.Linear(128, 768), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(fea_dim, 2)

        ## ARG
        # value为video
        self.co_attention_video_text = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads,
                                                    dropout=self.dropout,
                                                    d_model=fea_dim,
                                                    visual_len=fea_dim, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                                    pos=False)
        # 原本没dropout
        self.hard_mlp = nn.Sequential(nn.Linear(fea_dim, self.mlp_out_dim),
                                      nn.ReLU(),
                                      # nn.Dropout(p=self.dropout),
                                      nn.Linear(self.mlp_out_dim, 1),
                                      nn.Sigmoid()
                                      )
        # 原本没dropout
        self.simple_mlp = nn.Sequential(nn.Linear(fea_dim, self.mlp_out_dim),
                                        nn.ReLU(),
                                        # nn.Dropout(p=self.dropout),
                                        nn.Linear(self.mlp_out_dim, 3))

        self.simple_text_attention = MaskAttention(fea_dim)

        self.score_mapper = nn.Sequential(nn.Linear(fea_dim, self.mlp_out_dim),
                                          nn.BatchNorm1d(self.mlp_out_dim),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(self.mlp_out_dim, 64),
                                          nn.BatchNorm1d(64),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(64, 1),
                                          nn.Sigmoid()
                                          )

    def forward(self, **kwargs):
        #below codes are video fusion
        # pre_score
        pre_score = kwargs['pre_score']
        pre_score = pre_score.unsqueeze(1)

        # gpt_label
        gpt_label = kwargs['gpt_label']
        gpt_label = gpt_label.unsqueeze(1)

        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512)
        title_mask = kwargs['title_mask']  # (batch,512)

        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
        fea_text = self.linear_text(fea_text)

        ### Audio Frames ###
        audioframes = kwargs['audioframes']  # (batch,36,12288)
        fea_audio = self.vggish_modified(audioframes)  # (batch, frames, 128)
        # fea_audio = self.linear_project(fea_audio)
        fea_audio = self.linear_audio(fea_audio)
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1],
                                                   s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        ### Image Frames ###
        frames = kwargs['frames']  # (batch,30,4096)
        fea_img = self.linear_img(frames)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)


        ### C3D ###
        c3d = kwargs['c3d']  # (batch, 36, 4096)
        fea_video = self.linear_video(c3d)  # (batch, frames, 128)
        fea_video, fea_text = self.co_attention_tv(v=fea_video, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])

        fea_text = torch.mean(fea_text, -2)
        fea_video = torch.mean(fea_video, -2)

        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)

        a_1 = torch.tensor(0.1, requires_grad=True)
        a_2 = torch.tensor(0.01, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=True)
        reweight_pre_score = self.KIS(pre_score, a=a_1, b=b)


        fea = torch.cat((fea_text, fea_audio, fea_video, fea_img), 1)  # (bs, 4, 128)
        fea = self.trm(fea)
        video_fea = torch.mean(fea, -2)  # video_shape[128,768]
        video_fea_1, video_fea_2 = video_fea, video_fea

        # style
        style_inputid = kwargs['style_inputid']  # (batch,512)
        style_mask = kwargs['style_mask']  # (batch,512)
        fea_style = self.bert(style_inputid, attention_mask=style_mask)['last_hidden_state']
        fea_style = self.linear_text(fea_style)

        # content
        content_inputid = kwargs['content_inputid']  # (batch,512)
        content_mask = kwargs['content_mask']  # (batch,512)
        fea_content = self.bert(content_inputid, attention_mask=content_mask)['last_hidden_state']
        fea_content = self.linear_text(fea_content)

        # match
        match_inputid = kwargs['match_inputid']  # (batch,512)
        match_mask = kwargs['match_mask']  # (batch,512)
        fea_match = self.bert(match_inputid, attention_mask=match_mask)['last_hidden_state']
        fea_match = self.linear_text(fea_match)

        # below codes are VII&KIS
        fea_style_video, fea_video_style = self.co_attention_video_text(s=video_fea_2, v=fea_style,
                                                                        s_len=video_fea_2.shape[1],
                                                                        v_len=fea_style.shape[1])
        expert_2 = torch.mean(fea_style_video, dim=-2)

        fea_content_video, fea_video_content = self.co_attention_video_text(s=fea_video_style, v=fea_content,
                                                                            s_len=fea_video_style.shape[1],
                                                                            v_len=fea_content.shape[1])
        expert_3 = torch.mean(fea_content_video, dim=-2)

        fea_match_video, fea_video_match = self.co_attention_video_text(s=fea_video_content, v=fea_match,
                                                                        s_len=fea_video_content.shape[1],
                                                                        v_len=fea_match.shape[1])
        expert_4 = torch.mean(fea_match_video, dim=-2)

        expert_1 = torch.mean(fea_video_match, dim=-2)

        reweight_expert_2 = expert_2 * reweight_pre_score
        reweight_expert_3 = expert_3 * reweight_pre_score
        reweight_expert_4 = expert_4 * reweight_pre_score


        all_feature = torch.cat(
            (expert_1.unsqueeze(1), reweight_expert_2.unsqueeze(1),
             reweight_expert_3.unsqueeze(1), reweight_expert_4.unsqueeze(1)),
            dim=-2
        )
        all_feature = torch.mean(all_feature, dim=-2)
        output = self.classifier(all_feature)

        return output, all_feature

    def KIS(self, x, a, b):
        return 1 / (1 + torch.exp(-a * 100 * (x - self.fuse_threshold))) + b
