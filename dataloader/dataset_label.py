import json
import random
import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

import loader
from augment import VideoAug


class BaseDataset(Dataset):
    """Base Dataset"""

    def __init__(self, metafile):
        super().__init__()
        self.metafile = metafile        
        self.meta, self.indexes = self.parse_metafile(metafile)

    def parse_metafile(self, metafile):
        with open(metafile) as f:
            meta = json.load(f)
        return meta, list(meta.keys())

    def logging(self, logger):
        logger.info('metafile: {}'.format(self.metafile))
        logger.info('size: {}'.format(len(self.indexes)))


class VideoDataset(BaseDataset):
    """Video Dataset"""

    def __init__(self, metafile, max_length, training):
        super().__init__(metafile)
        self.max_length = max_length
        self.training = training
        self.video_transform = VideoAug()

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        videos, masks = loader.video_loader(video_info['frames'], 
                                            self.max_length, 
                                            self.training, 
                                            self.video_transform, 
                                            1)
        
        if self.training:
            return videos, masks
        return videos, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('max length: {}'.format(self.max_length))
        logger.info('training: {}'.format(self.training))


class TextDataset(BaseDataset):
    def __init__(self, meta_file, model_id, max_length, training, drop_rate):
        super().__init__(meta_file)

        self.max_length = max_length
        self.training = training
        self.drop_rate = drop_rate
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        text_info = self.meta[index]
        tokens, masks = loader.text_loader(text_info['caption'], 
                                           self.tokenizer, 
                                           self.max_length, 
                                           self.training, 
                                           1, 
                                           self.drop_rate)
                    
        if self.training:
            return tokens, masks
        return tokens, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('model id: {}'.format(self.model_id))
        logger.info('max length: {}'.format(self.max_length))
        logger.info('training: {}'.format(self.training))
        logger.info('drop rate: {}'.format(self.drop_rate))


class VTDatasetForDML(BaseDataset):
    def __init__(self, dataset_conf, video_conf, text_conf, training):
        super().__init__(dataset_conf['data_file'])
        self.labels = self.label_to_video_list()
        
        self.training = training
        self.nsamples = dataset_conf['nsamples']
        self.videoset = VideoDataset(dataset_conf['data_file'],
                                     video_conf['max_length'],
                                     training)
        self.textset = TextDataset(dataset_conf['data_file'],
                                   text_conf['model_id'],
                                   text_conf['max_length'],
                                   training,
                                   dataset_conf['text_drop_rate'])

    def label_to_video_list(self):
        self.label_meta = {}
        for k in self.meta.keys():
            for label in self.meta[k]['label']:
                if label not in self.label_meta.keys():
                    self.label_meta[label] = []
                self.label_meta[label].append(k)

        one_label = []
        for label in self.label_meta.keys():
            if len(self.label_meta[label]) <= 1:
                one_label.append(label)
        for label in one_label:
            del self.label_meta[label]

        return list(self.label_meta.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):  
        label = self.labels[index]
        
        if self.training:
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
        else:
            videos, vmasks, _ = self.videoset[index]
            texts, tmasks, _ = self.textset[index]
            return videos, vmasks, texts, tmasks, index

    def logging(self, logger):
        super().logging(logger)
        self.videoset.logging(logger)
        self.textset.logging(logger)
