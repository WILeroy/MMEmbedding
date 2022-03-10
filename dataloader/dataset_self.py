import json

from torch.utils.data.dataset import Dataset

import loader
from augment import RandomVideoAug


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

    def __len__(self):
        return len(self.indexes)

    def logging(self, logger):
        logger.info('metafile: {}'.format(self.metafile))
        logger.info('size: {}'.format(self.__len__()))


class VideoDataset(BaseDataset):
    """Video Dataset"""

    def __init__(self, metafile, max_length, training, transform_cnt=0):
        super().__init__(metafile)

        self.max_length = max_length
        self.training = training
        self.transform_cnt = transform_cnt

        self.video_transform = RandomVideoAug()

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
    
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        videos, masks = loader.video_loader(video_info['frames'], 
                                            self.max_length, 
                                            self.training, 
                                            self.video_transform, 
                                            self.transform_cnt)
        
        if self.training:
            return videos, masks
        return videos, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('max length: {}'.format(self.max_length))
        logger.info('training: {}'.format(self.training))
        logger.info('transform count: {}'.format(self.transform_cnt))
        

class AudioDataset(BaseDataset):
    def __init__(self, meta_file, max_length, training, transform_cnt=0, drop_rate=0):
        super().__init__(meta_file)

        self.max_length = max_length
        self.training = training
        self.transform_cnt = transform_cnt
        self.drop_rate = drop_rate

    def __getitem__(self, index):
        assert isinstance(index, (int, str))

        if isinstance(index, int):
            index = self.indexes[index]
    
        audio_info = self.meta[index]
        audios, masks = loader.audio_loader(audio_info['mel'], 
                                            self.max_length, 
                                            self.training, 
                                            self.transform_cnt,
                                            self.drop_rate)
        
        if self.training:
            return audios, masks
        return audios, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('max length: {}'.format(self.max_length))
        logger.info('training: {}'.format(self.training))
        logger.info('transform count: {}'.format(self.transform_cnt))
        logger.info('drop rate: {}'.format(self.drop_rate))


class VADataset(BaseDataset):
    def __init__(self, dataset_conf, video_conf, audio_conf, training):
        super().__init__(dataset_conf['data_file'])

        self.training = training

        self.videoset = VideoDataset(dataset_conf['data_file'],
                                     video_conf['max_length'],
                                     training,
                                     dataset_conf['transform_cnt'])
        self.audioset = AudioDataset(dataset_conf['data_file'],
                                     audio_conf['max_length'],
                                     training,
                                     dataset_conf['transform_cnt'],
                                     dataset_conf['audio_drop_rate'])

    def __getitem__(self, index):  
        assert isinstance(index, (int, str))
    
        if isinstance(index, int):
            index = self.indexes[index]
        
        if self.training:
            videos, vmasks = self.videoset[index]
            audios, amasks = self.audioset[index]
            return videos, vmasks, audios, amasks
        else:
            videos, vmasks, vindex = self.videoset[index]
            audios, amasks, aindex = self.audioset[index]
            return videos, vmasks, audios, amasks, index

    def logging(self, logger):
        super().logging(logger)
        self.videoset.logging(logger)
        self.audioset.logging(logger)
