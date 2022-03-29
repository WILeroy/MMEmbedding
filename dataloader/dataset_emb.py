from transformers import AutoTokenizer
from .dataset_base import BaseDataset
from . import loader


class VideoDatasetEmbedding(BaseDataset):
    def __init__(self, meta, max_num_frames, **kwargs):
        super().__init__(meta, **kwargs)

        self.max_num_frames = max_num_frames

    def __getitem__(self, index):
        assert isinstance(index, (int, str))
        if isinstance(index, int):
            index = self.indexes[index]
    
        video_info = self.meta[index]
        videos, vmasks = loader.video_loader(frames_path=video_info['frames'], 
                                             max_length=self.max_num_frames, 
                                             training=False, 
                                             video_transform=None)
        
        return videos, vmasks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('VideoDatasetEmbedding max_num_frames: {}'.format(self.max_num_frames))


class TextDatasetEmbedding(BaseDataset):
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
        tokens, masks = loader.text_loader(caption=text_info['caption'], 
                                           tokenizer=self.tokenizer, 
                                           max_length=self.max_num_tokens, 
                                           training=False)

        return tokens, masks, index

    def logging(self, logger):
        super().logging(logger)
        logger.info('TextDatasetEmbedding tokenizer_id: {}'.format(self.tokenizer_id))
        logger.info('TextDatasetEmbedding max_num_tokens: {}'.format(self.max_num_tokens))


class VTDatasetEmbedding(VideoDatasetEmbedding, TextDatasetEmbedding):
    def __init__(self, meta, max_num_frames, tokenizer_id, max_num_tokens):
        super().__init__(meta, 
                         max_num_frames=max_num_frames, 
                         tokenizer_id=tokenizer_id, 
                         max_num_tokens=max_num_tokens)

    def __getitem__(self, index):
        videos, vmasks, vindex = VideoDatasetEmbedding.__getitem__(self, index)
        tokens, tmasks, tindex = TextDatasetEmbedding.__getitem__(self, index)
        assert vindex == tindex

        return videos, vmasks, tokens, tmasks, vindex
    
    def logging(self, logger):
        super().logging(logger)
