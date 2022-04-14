import json
import random
import torch
from torch.utils.data.dataset import Dataset

from .augment import VideoAug
from .dataset_base import VideoBaseDataset, TextBaseDataset


class VTDatasetLabel(Dataset):
    def __init__(self, dataset_conf, video_conf, text_conf):
        super().__init__()
        self.labels, self.label_meta = self.parse_meta(dataset_conf['data_file'])
        
        self.nsamples = dataset_conf['nsamples']
        self.videoset = VideoBaseDataset(
            meta = dataset_conf['data_file'],
            max_num_frames = video_conf['max_length'],
            training = True,
            transform = VideoAug(),
            transform_cnt = 1
        )
        self.textset = TextBaseDataset(
            meta = dataset_conf['data_file'],
            tokenizer_id = text_conf['model_id'],
            max_num_tokens = text_conf['max_length'],
            training = True,
            drop_rate = dataset_conf['text_drop_rate'],
            transform_cnt = 1
        )

    def parse_meta(self, meta):
        if isinstance(meta, str):
            with open(meta) as f:
                meta = json.load(f)

        label_meta = {}
        for k in meta.keys():
            for label in meta[k]['label']:
                if label not in label_meta.keys():
                    label_meta[label] = []
                label_meta[label].append(k)

        one_label = []
        for label in label_meta.keys():
            if len(label_meta[label]) <= 1:
                one_label.append(label)
        for label in one_label:
            del label_meta[label]

        return list(label_meta.keys()), label_meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):  
        label = self.labels[index]
        videoids = random.sample(self.label_meta[label], self.nsamples)

        videos, vmasks, texts, tmasks, labels = [], [], [], [], []
        for videoid in videoids:
            video, vmask = self.videoset[videoid]
            text, tmask = self.textset[videoid]
            videos.append(video)
            vmasks.append(vmask)
            texts.append(text)
            tmasks.append(tmask)
            labels.append(label)
                
        return torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), torch.cat(texts, dim=0), torch.cat(tmasks, dim=0), torch.tensor(labels)
        