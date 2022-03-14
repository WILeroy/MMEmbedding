import argparse
import json
import os
import os.path as osp
import shutil

import numpy as np
import torch
import torch.multiprocessing as mp
import utils.utils as utils
from dataloader.partition import DataPartitioner
from trainer.trainer_stam_sbert import TrainerStamSbert

os.environ["CUDA_VISIBLE_DEVICES"] = '8,9,10,11,12,13,14,15'


def run(rank, config, trainset):
    trainer = TrainerStamSbert(config)
    trainer.set_rank(rank, config['ngpus'])
    trainer.train()


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

    trainset = VTDatasetLabel(config['trainset'], config['experts']['stam'], config['experts']['sbert'], True)
    trainset.logging(logger)
    
    # Split dataset.
    config['ngpus'] = torch.cuda.device_count()
    size_gpus = [config['train']['size_per_gpu'] for _ in range(config['ngpus'])]
    partitioner = DataPartitioner(trainset, np.array(size_gpus)/np.sum(size_gpus), True)

    # Running.
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(run, nprocs=config['ngpus'], args=(config, partitioner))
