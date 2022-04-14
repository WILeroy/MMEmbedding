import numpy as np
import torch
import torch.distributed as dist
import json


class Merger():
    def __init__(self, meta):
        if isinstance(meta, str):
            with open(meta) as f:
                meta = json.load(f)
        self.meta = meta

    def merge(self, multi_labels):
        """
        args:
            multi_labels: [[str]]
    
        returns:
            labels
        """
        num_item = len(multi_labels)
        inter_mat = np.zeros((num_item, num_item))
        for h in range(num_item):
            for w in range(num_item):
                inter = list(set(multi_labels[h]).intersection(set(multi_labels[w])))
                inter_mat[h, w] = len(inter)

        labels = [0 for i in range(num_item)]
        next_label = 1
        for h in range(num_item):
            if labels[h] != 0: continue

            tmp_list = [h]
            res_list = []
            visited = [False if labels[i] == 0 else True for i in range(num_item)]

            while len(tmp_list) != 0:
                anchor = tmp_list[0]
                tmp_list.pop(0)
                visited[anchor] = True

                pos_indexes = np.where(inter_mat[anchor] >= 1)[0]
                for pos in pos_indexes:

                    if not visited[pos]:
                        tmp_list.append(pos)
                        visited[pos] = True
            
                res_list.append(anchor)

            for res in res_list:
                labels[res] = next_label
        
            next_label += 1

        return labels

    def __call__(self, rank, world_size, x):
        bsize = x.size()[0]
        x_list = [torch.ones_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x.contiguous())
        indexes = torch.cat(x_list, dim=0)
        indexes = indexes.detach().cpu().numpy().tolist()

        multi_labels = []
        for index in indexes:
            idx = str(index)
            multi_labels.append(self.meta[idx]['label'])

        labels = self.merge(multi_labels)
        labels = torch.tensor(labels[rank*bsize:(rank+1)*bsize], dtype=torch.int32)
        return labels