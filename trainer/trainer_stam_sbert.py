import functools

import torch
import torch.distributed as dist
import pytorch_metric_learning.utils.distributed as pml_dist
import torch.optim as optim
from model import MLPFusion, TextExpert, VideoExpert
from pytorch_metric_learning import distances, losses, miners, reducers
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.lr_scheduler import warm_up_with_cosine
from dataloader.dataset_label import VTDatasetForDML
from dataloader.dataset_self import VTDataset
from .trainer_base import TrainerBase
from torch.utils.data import DataLoader
import dataloader.collate as collate


class TrainerStamSbert(TrainerBase):
    def __init__(self, config):
        super().__init__(config)

        self.stam_config = config['experts']['stam']
        self.sbert_config = config['experts']['sbert']
        self.fusion_config = config['fusion']

        self.trainset_config = config['trainset']

        self.epochs = config['train']['epochs']
        self.warm_up_epochs = config['train']['warm_up_epochs']
        self.lr_init = config['train']['lr_init']

        self.set_model()
        self.set_trainset()
        self.set_valset()
        self.set_loss_function()
        self.set_optimizer()

    def set_model(self):
        stam = VideoExpert(self.stam_config)
        sbert = TextExpert(self.sbert_config)
        if self.fusion_config['name'] == 'fc':
            fusion = MLPFusion(
                self.stam_config['dim']+self.sbert_config['dim'], 
                self.fusion_config['dim'])

        if self.config['resume'] is not None:        
            stam.load_state_dict(self.state_dict['stam'])
            sbert.load_state_dict(self.state_dict['sbert'])
            fusion.load_state_dict(self.state_dict['fusion'])

        self.stam = DDP(stam.to(self.device), device_ids=[self.rank])
        self.sbert = DDP(sbert.to(self.device), device_ids=[self.rank], find_unused_parameters=True)
        self.fusion = DDP(fusion.to(self.device), device_ids=[self.rank])

    def save_model(self, epoch):
        state = {'epoch': epoch,
                 'stam': self.stam.module.state_dict(),
                 'sbert': self.sbert.module.state_dict(),
                 'fusion': self.fusion.module.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        return state

    def set_trainset(self):
        dataset = VTDatasetForDML(self.trainset_config, self.stam_config, self.sbert_config)
        self.trainloader = DataLoader(dataset,
                                      batch_size=self.trainset_config['size_per_gpu'], 
                                      shuffle=True,
                                      num_workers=6,
                                      collate_fn=collate.vtset_train_collate,
                                      drop_last=True)

    def set_valset(self):
        dataset = VTDataset(self.trainset_config, self.stam_config, self.sbert_config, False)
        self.valloader = DataLoader(dataset,
                                    batch_size=self.trainset_config['size_per_gpu']*3,
                                    collate_fn=collate.vtset_emb_collate,
                                    num_workers=6)

    def set_loss_function(self):
        # Set distributed loss function.
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_func = losses.TripletMarginLoss(margin=0.3, distance=distance, reducer=reducer)
        self.loss_func = pml_dist.DistributedLossWrapper(loss=loss_func, efficient=True)

        # Set distributed miner function.
        miner_func = miners.TripletMarginMiner(
            margin=0.3, distance=distance, type_of_triplets='all')
        self.miner_func = pml_dist.DistributedMinerWrapper(miner=miner_func, efficient=True)

    def set_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.stam.parameters()}, 
                                     {'params': self.sbert.parameters()},
                                     {'params': self.fusion.parameters()}
            ], lr=self.lr_init)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, 
            functools.partial(warm_up_with_cosine, 
                              warm_up_epochs=self.warm_up_epochs, 
                              total_epochs=self.epochs))

    def train(self):
        self.stam.train()
        self.sbert.train()
        self.fusion.train()

        for videos, vmasks, texts, tmasks, labels in self.trainloader:
            videos, vmasks = videos.to(self.device), vmasks.to(self.device)
            texts, tmasks = texts.to(self.device), tmasks.to(self.device)
            labels = labels.to(self.device)

            vembeddings = self.stam(videos, vmasks)['pooled_feature']
            tembeddings = self.sbert(texts, tmasks)['pooled_feature']
            embeddings = self.fusion(torch.cat([vembeddings, tembeddings], dim=1))
            dist.barrier()

            self.optimizer.zero_grad()
            indices_tuple = self.miner_func(embeddings, labels)
            loss = self.loss_func(embeddings, labels, indices_tuple)
            loss.backward()
            self.optimizer.step()

    def val(self):
        self.stam.val()
        self.sbert.val()
        self.fusion.val()

        for videos, vmasks, texts, tmasks, vids in self.valloader:
            videos, vmasks = videos.to(self.device), vmasks.to(self.device)
            texts, tmasks = texts.to(self.device), tmasks.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                vembeddings = self.stam(videos, vmasks)['pooled_feature']
                tembeddings = self.sbert(texts, tmasks)['pooled_feature']
                embeddings = self.fusion(torch.cat([vembeddings, tembeddings], dim=1))

            for i, vid in enumerate(vids):
                pass

        # ToDO: evaluate 