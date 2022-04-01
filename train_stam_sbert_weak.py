import argparse
import functools
import json
import os
import os.path as osp
import shutil
import time

import numpy as np
import pytorch_metric_learning.utils.distributed as pml_dist
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataloader.collate as collate
import utils.utils as utils
from dataloader.dataset_label import VTDatasetLabel
from dataloader.partition import DataPartitioner
from model.mlp_fusion import MLPFusion
from model.text_expert import TextExpert
from model.video_expert import VideoExpert
from utils.lr_scheduler import warm_up_with_cosine


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7'


def run(rank, config, trainset):
    logger = utils.define_logger(config['logdir'], 'train.log')
    logger.info("Rank {} entering the 'run' function".format(rank))
    if rank == 0:
        writer = SummaryWriter(config['logdir'])

    # Init rank.
    dist.init_process_group('nccl', rank=rank, world_size=config['ngpus'])
    torch.cuda.set_device(rank) # !!! important !!!
    device = torch.device(rank)
    
    # Get splited dataset.
    partition = trainset.use(rank)
    partition.logging(logger)

    dataloader = DataLoader(partition,
                            batch_size=config['train']['size_per_gpu'], 
                            shuffle=True,
                            num_workers=12,
                            collate_fn=collate.vtset_train_collate,
                            drop_last=True)

    # Resume.
    if config['resume'] is not None:
        config['experts']['stam']['pretrain'] = False
        state_dict = torch.load(config['resume'], map_location='cpu')

        logger.info('rank {}, resume: {}'.format(rank, config['resume']))
        logger.info('rank {}, latest epoch: {}'.format(rank, state_dict['epoch']))

    # Create model and set to DDP.
    stam = VideoExpert(config['experts']['stam'])
    sbert = TextExpert(config['experts']['sbert'])
    if config['fusion']['name'] == 'fc':
        fusion = MLPFusion(
            config['experts']['stam']['dim']+config['experts']['sbert']['dim'], 
            config['fusion']['dim'])
    
    stam.logging(logger)
    sbert.logging(logger)

    if config['resume'] is not None:        
        stam.load_state_dict(state_dict['stam'])
        sbert.load_state_dict(state_dict['sbert'])
        fusion.load_state_dict(state_dict['fusion'])

    stam = DDP(stam.to(device), device_ids=[rank])
    sbert = DDP(sbert.to(device), device_ids=[rank], find_unused_parameters=True)
    fusion = DDP(fusion.to(device), device_ids=[rank])

    # Set optimizer.
    optimizer = optim.Adam([{'params': stam.parameters()}, 
                            {'params': sbert.parameters()},
                            {'params': fusion.parameters()}
        ], lr=config['train']['lr_init'])
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        functools.partial(warm_up_with_cosine, 
                          warm_up_epochs=config['train']['warm_up_epochs'], 
                          total_epochs=config['train']['epochs']))

    if config['resume'] is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['scheduler'])
    latest_epoch = state_dict['epoch'] if config['resume'] is not None else -1
    start_epoch = latest_epoch + 1
    
    # Set distributed loss function.
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.3, distance=distance, reducer=reducer)
    loss_func = pml_dist.DistributedLossWrapper(loss=loss_func, efficient=True)

    # Set distributed miner function.
    miner_func = miners.TripletMarginMiner(
        margin=0.3, distance=distance, type_of_triplets='all')
    miner_func = pml_dist.DistributedMinerWrapper(miner=miner_func, efficient=True)

    # Training.
    global_step = 1
    loss_1k = 0
    t = time.time()

    for epoch in range(start_epoch, config['train']['epochs']):
        logger.info("Rank {} starting epoch {}".format(rank, epoch))
        logger.info("Total steps = {}".format(len(dataloader)))
        logger.info("Learning rate = {:.7f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        t_epoch = time.time()

        for videos, vmasks, texts, tmasks, labels in dataloader:
            videos, vmasks = videos.to(device), vmasks.to(device)
            texts, tmasks = texts.to(device), tmasks.to(device)
            labels = labels.to(device)

            vembeddings = stam(videos, vmasks)['pooled_feature']
            tembeddings = sbert(texts, tmasks)['pooled_feature']
            embeddings = fusion(torch.cat([vembeddings, tembeddings], dim=1))
            dist.barrier()

            optimizer.zero_grad()

            indices_tuple = miner_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            loss.backward()

            optimizer.step()
            
            if rank == 0:
                writer.add_scalar('loss@rank0', loss.item(), global_step)

            loss_1k += loss.item()
            if global_step % 1000 == 0:
                used_t = time.time() - t
                logger.info("Rank {}, iteration {} k, loss {:.6f}, used {:.2f} s/k".format(
                    rank, int(global_step/1000), loss_1k/1000, used_t))
                loss_1k = 0
                t = time.time()
            global_step += 1
        
        logger.info("Rank {}, epoch {}, used {:.2f} min/epoch".format(
            rank, epoch, (time.time()-t_epoch)/60))

        dist.barrier()
        lr_scheduler.step()

        if rank == 0 and (epoch) % config['train']['save_interval'] == 0:
            state = {'epoch': epoch,
                     'stam': stam.module.state_dict(),
                     'sbert': sbert.module.state_dict(),
                     'fusion': fusion.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': lr_scheduler.state_dict()}
            torch.save(state, osp.join(config['logdir'], f'model_{epoch}.pth'))


def parse_config(args):
    with open(args.config_file) as f: 
        config = json.load(f)

    config['resume'] = args.resume
    config['logdir'] = osp.join(config['logdir'], config['expname'])
    if config['resume'] is None:
        if osp.exists(config['logdir']): shutil.rmtree(config['logdir'])
        os.makedirs(config['logdir'])
    os.makedirs(config['logdir'], exist_ok=True)

    # save config file to logdir.
    os.system('cp {} {}'.format(args.config_file, config['logdir']))

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--port', type=str, default='23333')
    parser.add_argument('--seed', type=int, default=23333)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    utils.set_random_seed(seed=args.seed)
    config = parse_config(args)
    logger = utils.define_logger(config['logdir'], 'train.log')

    trainset = VTDatasetLabel(config['trainset'], config['experts']['stam'], config['experts']['sbert'])
    trainset.logging(logger)
    
    # Split dataset.
    config['ngpus'] = torch.cuda.device_count()
    size_gpus = [config['train']['size_per_gpu'] for _ in range(config['ngpus'])]
    partitioner = DataPartitioner(trainset, np.array(size_gpus)/np.sum(size_gpus), True)

    # Running.
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(run, nprocs=config['ngpus'], args=(config, partitioner))
