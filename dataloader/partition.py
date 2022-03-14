import json
import random
        

class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    def __init__(self, data_file, sizes, training):
        self.meta, self.indexes = self.parse_metafile(data_file)
        self.partitions = []
        self.training = training

        data_len = len(self.indexes)
        if training: random.Random().shuffle(self.indexes)

        for part_to_rank, frac in enumerate(sizes):
            if not training:
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
