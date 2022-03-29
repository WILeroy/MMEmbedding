import json

from torch.utils.data.dataset import Dataset


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
