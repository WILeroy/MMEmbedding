import random
import json


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index, training):
        self.data = data
        self.index = index
        self.training = training

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        item = self.data[data_idx]
        
        return item

    def logging(self, logger):
        logger.info('Partition size: {}'.format(self.__len__()))
        

class DataPartitioner(object):
    """Partitions a dataset (from torch-like dataset) into different chuncks."""

    def __init__(self, data, sizes, training):
        self.data = data
        self.partitions = []
        self.training = training

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        if training: random.Random().shuffle(indexes)

        for part_to_rank, frac in enumerate(sizes):
            if not training:
                if part_to_rank == (len(sizes) - 1):
                    self.partitions.append(indexes)
                    continue
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition], self.training)


class MetaPartitioner(object):
    """Partitions a dataset (from metafile) into different chuncks."""

    def __init__(self, data_file, sizes):
        self.meta, self.indexes = self.parse_metafile(data_file)
        self.partitions = []

        data_len = len(self.indexes)

        for part_to_rank, frac in enumerate(sizes):
            if part_to_rank == (len(sizes) - 1):
                self.partitions.append(self.indexes)
                continue
            part_len = int(frac * data_len)
            self.partitions.append(self.indexes[0:part_len])
            self.indexes = self.indexes[part_len:]

    def parse_metafile(self, metafile):
        with open(metafile) as f:
            meta = json.load(f)
        return meta, list(meta.keys())

    def use(self, partition):
        return {index:self.meta[index] for index in self.partitions[partition]}