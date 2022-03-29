import json
import random

import torch
from transformers import AutoTokenizer

from . import loader
from .augment import VideoAug
from .dataset_base import BaseDataset


class VideoDatasetBase(BaseDataset):
    """Based Video Dataset, use video_id to load frames and masks."""

    def __init__(self, meta, max_num_frames, **kwargs):
        super().__init__(meta, **kwargs)
        self.max_num_frames = max_num_frames
        self.video_transform = VideoAug()

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        frames, masks = loader.video_loader(frames_path=video_info['frames'], 
                                            max_length=self.max_num_frames, 
                                            training=True, 
                                            video_transform=self.video_transform)
        
        return frames, masks

    def logging(self, logger):
        super().logging(logger)
        logger.info('VideoDatasetBase max_num_frames: {}'.format(self.max_num_frames))


class TextDatasetBase(BaseDataset):
    """Based Text Dataset, use video_id to load toekn_ids and masks."""

    def __init__(self, meta, tokenizer_id, max_num_tokens, drop_rate, **kwargs):
        super().__init__(meta, **kwargs)

        self.tokenizer_id = tokenizer_id
        self.max_num_tokens = max_num_tokens
        self.drop_rate = drop_rate
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        text_info = self.meta[index]
        tokens, masks = loader.text_loader(caption=text_info['caption'], 
                                           tokenizer=self.tokenizer, 
                                           max_length=self.max_num_tokens, 
                                           training=False, 
                                           drop_rate=self.drop_rate)
                    
        return tokens, masks

    def logging(self, logger):
        super().logging(logger)
        logger.info('model id: {}'.format(self.tokenizer_id))
        logger.info('max length: {}'.format(self.max_num_tokens))
        logger.info('drop rate: {}'.format(self.drop_rate))


class VTDatasetLabel(VideoDatasetBase, TextDatasetBase):
    def __init__(self, dataset_conf, video_conf, text_conf):
        super().__init__(dataset_conf['data_file'], 
                         max_num_frames=video_conf['max_length'], 
                         tokenizer_id=text_conf['model_id'],
                         max_num_tokens=text_conf['max_length'],
                         drop_rate=dataset_conf['text_drop_rate'])

        self.label_meta, self.labels = self.label_to_video_list()
        self.nsamples = dataset_conf['nsamples']

    def label_to_video_list(self):
        label_meta = {}
        for k in self.meta.keys():
            for label in self.meta[k]['label']:
                if label not in label_meta.keys():
                    label_meta[label] = []
                label_meta[label].append(k)

        one_label = []
        for label in label_meta.keys():
            if len(label_meta[label]) <= 1:
                one_label.append(label)
        for label in one_label:
            del label_meta[label]

        return label_meta, list(label_meta.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):  
        label = self.labels[index]
        
        videoids = random.sample(self.label_meta[label], self.nsamples)

        videos, vmasks, texts, tmasks, labels = [], [], [], [], []
        for videoid in videoids:
            video, vmask = VideoDatasetBase.__getitem__(self, videoid)
            text, tmask = TextDatasetBase.__getitem__(self, videoid)
            videos.append(video)
            vmasks.append(vmask)
            texts.append(text)
            tmasks.append(tmask)
            labels.append(label)
        
        return (torch.cat(videos, dim=0), 
                torch.cat(vmasks, dim=0), 
                torch.cat(texts, dim=0), 
                torch.cat(tmasks, dim=0), 
                torch.tensor(labels))
        
    def logging(self, logger):
        super().logging(logger)
        logger.info('VTDatasetLabel nsamples: {}'.format(self.nsamples))