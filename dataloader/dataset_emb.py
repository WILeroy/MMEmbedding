import json
from torch.utils.data.dataset import Dataset

from .dataset_base import TextBaseDataset, VideoBaseDataset


class VTDatasetEmbedding(Dataset):
    def __init__(self, meta, video_conf, text_conf):
        super().__init__()
        self.indexes = self.parse_meta(meta)

        self.videoset = VideoBaseDataset(
            meta = meta,
            max_num_frames = video_conf['max_length'],
            training = False,
            transform = None,
            transform_cnt = 1
        )
        self.textset = TextBaseDataset(
            meta = meta,
            tokenizer_id = text_conf['model_id'],
            max_num_tokens = text_conf['max_length'],
            training = False,
            drop_rate = 0,
            transform_cnt = 1
        )

    def parse_meta(self, meta):
        if isinstance(meta, str):
            with open(meta) as f:
                meta = json.load(f)
        
        return list(meta.keys())

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        videoid = self.indexes[index]
        videos, vmasks, vindex = self.videoset[videoid]
        tokens, tmasks, tindex = self.textset[videoid]
        assert vindex == tindex

        return videos, vmasks, tokens, tmasks, vindex
    
    def logging(self, logger):
        logger.info('VTDatasetEmbedding size: {}'.format(self.__len__()))
        self.videoset.logging(logger)
        self.textset.logging(logger)
