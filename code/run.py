import collections
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from gensim.models import KeyedVectors

from models.IFAIModel import MainModel

from utils.dataloader import *
from models.Trainer import Trainer


def pad_sequence(seq_len, lst, emb):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)


def pad_sequence_bbox(seq_len, lst):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, 45, 4096], dtype=torch.float)
        elif ori_len >= seq_len:
            video = torch.FloatTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, 45, 4096], dtype=torch.float)], dim=0)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def _init_fn(worker_id):
    np.random.seed(3407)


def IFAI_collate_fn(batch):
    num_frames = 83
    num_audioframes = 50

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    style_inputid = [item['style_inputid'] for item in batch]
    style_mask = [item['style_mask'] for item in batch]

    content_inputid = [item['content_inputid'] for item in batch]
    content_mask = [item['content_mask'] for item in batch]

    match_inputid = [item['match_inputid'] for item in batch]
    match_mask = [item['match_mask'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]

    gpt_label = [item['gpt_label'] for item in batch]

    pre_score = [item['pre_score'] for item in batch]

    return {
        'label': torch.stack(label),
        'gpt_label': torch.stack(gpt_label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'style_inputid': torch.stack(style_inputid),
        'style_mask': torch.stack(style_mask),
        'content_inputid': torch.stack(content_inputid),
        'content_mask': torch.stack(content_mask),
        'match_inputid': torch.stack(match_inputid),
        'match_mask': torch.stack(match_mask),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        'pre_score': torch.stack(pre_score)
    }


class Run():
    def __init__(self,
                 config
                 ):

        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'IFAI'

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd = config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616
        self.mode = 'normal'

    def get_dataloader(self, data_type, data_fold):
        collate_fn = None

        if data_type == 'IFAI':
            dataset_train = MPDataset(f'vid_fold_no_{data_fold}.txt')
            dataset_test = MPDataset(f'vid_fold_{data_fold}.txt')
            collate_fn =IFAI_collate_fn

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)

        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        dataloaders = dict(zip(['train', 'test'], [train_dataloader, test_dataloader]))

        return dataloaders



    def get_model(self):
        self.model = MainModel(bert_model='./code/models/bert-base-chinese',
                                   fea_dim=128, dropout=self.dropout)

        return self.model

    def main(self):
        if self.mode_eval == "nocv":
            self.model = self.get_model()
            dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=self.fold)
            trainer = Trainer(model=self.model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                              epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                              mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                              epoch_stop=self.epoch_stop,
                              save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                              writer=SummaryWriter(self.path_tensorboard))
            result = trainer.train()

        elif self.mode_eval == "temporal":
            self.model = self.get_model()
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type)
            trainer = Trainer3(model=self.model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                               epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                               mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                               epoch_stop=self.epoch_stop,
                               save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                               writer=SummaryWriter(self.path_tensorboard))
            result = trainer.train()
            return result

        elif self.mode_eval == "cv":
            collate_fn = None
            # if self.model_name == 'TextCNN':
            #     wv_from_text = KeyedVectors.load_word2vec_format(
            #         "./stores/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt",
            #         binary=False)

            history = collections.defaultdict(list)
            for fold in range(1, 6):
                print('-' * 50)
                print('fold %d:' % fold)
                print('-' * 50)
                self.model = self.get_model()
                dataloaders = self.get_dataloader(data_type=self.data_type, data_fold=fold)
                trainer = Trainer(model=self.model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                                  epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                                  mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                                  epoch_stop=self.epoch_stop,
                                  save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                                  writer=SummaryWriter(self.path_tensorboard + "fold_" + str(fold) + "/"))

                result = trainer.train()

                history['auc'].append(result['auc'])
                history['f1'].append(result['f1'])
                history['recall'].append(result['recall'])
                history['precision'].append(result['precision'])
                history['acc'].append(result['acc'])

            print('results on 5-fold cross-validation: ')
            for metric in ['acc', 'f1', 'precision', 'recall', 'auc']:
                print('%s : %.4f +/- %.4f' % (metric, np.mean(history[metric]), np.std(history[metric])))

        else:
            print("Not Available")
