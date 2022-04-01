import json

from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from . import loader


class BaseDataset(Dataset):
    def __init__(self, meta):
        super().__init__()

        if isinstance(meta, dict):
            self.meta = meta
            self.indexes = list(meta.keys())
        elif isinstance(meta, str):
            self.meta, self.indexes = self.parse_metafile(meta)

    def parse_metafile(self, metafile):
        with open(metafile) as f:
            meta = json.load(f)
        return meta, list(meta.keys())

    def __len__(self):
        return len(self.indexes)

    def logging(self, logger):
        logger.info('BaseDataset size: {}'.format(self.__len__()))


class VideoBaseDataset(BaseDataset):
    """Based Dataset to load frames and mask."""

    def __init__(self, meta, max_num_frames, training, transform, transform_cnt):
        super().__init__(meta)
        self.max_num_frames = max_num_frames
        self.training = training
        self.video_transform = transform
        self.transform_cnt = transform_cnt

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        videos, masks = loader.video_loader(
            frames_path = video_info['frames'], 
            max_length = self.max_num_frames, 
            training = self.training, 
            video_transform = self.video_transform, 
            transform_cnt = self.transform_cnt
        )
        
        ret = (videos, masks) if self.training else (videos, masks, index)
        return ret

    def logging(self, logger):
        super().logging(logger)
        logger.info('VideoBaseDataset max_num_frames: {}'.format(self.max_num_frames))
        logger.info('VideoBaseDataset training: {}'.format(self.training))


class TextBaseDataset(BaseDataset):
    """Based Dataset to load tokens-ids and mask."""

    def __init__(self, meta, tokenizer_id, max_num_tokens, training, drop_rate, transform_cnt):
        super().__init__(meta)

        self.tokenizer_id = tokenizer_id
        self.max_num_tokens = max_num_tokens
        self.training = training
        self.drop_rate = drop_rate
        self.transform_cnt = transform_cnt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        text_info = self.meta[index]
        tokens, masks = loader.text_loader(
            caption = text_info['caption'], 
            tokenizer = self.tokenizer, 
            max_length = self.max_num_tokens, 
            training = self.training, 
            drop_rate = self.drop_rate,
            transform_cnt = self.transform_cnt
        )
                    
        ret = (tokens, masks) if self.training else (tokens, masks, index)
        return ret

    def logging(self, logger):
        super().logging(logger)
        logger.info('TextBaseDataset tokenizer_id: {}'.format(self.tokenizer_id))
        logger.info('TextBaseDataset max_num_tokens: {}'.format(self.max_num_tokens))
        logger.info('TextBaseDataset training: {}'.format(self.training))
        logger.info('TextBaseDataset drop_rate: {}'.format(self.drop_rate))
        logger.info('TextBaseDataset transform_cnt: {}'.format(self.transform_cnt))