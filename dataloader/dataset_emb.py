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
        logger.info('size: {}'.format(self.__len__()))


class VideoDataset(BaseDataset):
    def __init__(self, meta, max_num_frames, **kwargs):
        super().__init__(meta, **kwargs)

        self.max_num_frames = max_num_frames

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        videos, vmasks = loader.video_loader(video_info['frames'], self.max_num_frames, False, None, 1)
        
        return videos, vmasks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('max_num_frames: {}'.format(self.max_num_frames))


class TextDataset(BaseDataset):
    def __init__(self, meta, tokenizer_id, max_num_tokens, **kwargs):
        super().__init__(meta, **kwargs)
        self.tokenizer_id = tokenizer_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.max_num_tokens = max_num_tokens

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        text_info = self.meta[index]
        tokens, masks = loader.text_loader(
            text_info['caption'], self.tokenizer, self.max_num_tokens, False, 1, 0)

        return tokens, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('tokenizer_id: {}'.format(self.tokenizer_id))
        logger.info('max_num_tokens: {}'.format(self.max_num_tokens))


class VTDataset(VideoDataset, TextDataset):
    def __init__(self, meta, max_num_frames, tokenizer_id, max_num_tokens):
        super().__init__(meta, 
                         max_num_frames=max_num_frames, 
                         tokenizer_id=tokenizer_id, 
                         max_num_tokens=max_num_tokens)

    def __getitem__(self, index):
        frames, vmasks, vindex = VideoDataset.__getitem__(self, index)
        tokens, tmasks, tindex = TextDataset.__getitem__(self, index)
        assert vindex == tindex

        return frames, vmasks, tokens, tmasks, vindex
    
    def logging(self, logger):
        super().logging(logger)