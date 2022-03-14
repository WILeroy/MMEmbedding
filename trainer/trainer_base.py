import abc
import os

import torch
import torch.distributed as dist


class TrainerBase(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.epochs = config['train']['epochs']
        self.val_interval = config['train']['val_interval']
        self.save_interval = config['train']['save_interval']
        
    def set_dist_rank(self, backend, rank, wsize):
        dist.init_process_group(backend, rank=rank, world_size=wsize)
        torch.cuda.set_device(rank) # !!! important !!!
        self.rank = rank
        self.device = torch.device(rank)

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def val(self):
        raise NotImplementedError

    def run(self):
        """ Main function. """

        for epoch in self.epochs:
            self.train()
            
            if self.val_interval != 0 and epoch % self.val_interval == 0:
                self.val()

            