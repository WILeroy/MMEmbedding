import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import dataloader.collate as collate
import utils.utils as utils
from dataloader.dataset_emb import VTDatasetEmbedding
from model import MLPFusion, TextExpert, VideoExpert


def run(rank, config, partitioner):
    logger = utils.define_logger(config['logdir'], 'emb.log')
    logger.info("Rank {} entering the 'run' function".format(rank))
    
    # Init rank.
    torch.cuda.set_device(rank) # !!! important !!!
    device = torch.device(rank)

    # Get splited dataset.
    partition = partitioner.use(rank)

    dataset = VTDatasetEmbedding(partition, 
                        config['experts']['stam']['max_length'], 
                        config['experts']['sbert']['model_id'], 
                        config['experts']['sbert']['max_length'])
    dataset.logging(logger)
    dataloader = DataLoader(dataset,
                            batch_size=config['train']['size_per_gpu']*3,
                            collate_fn=collate.vtset_emb_collate,
                            num_workers=12)
    
    # Create model and load weights.
    model_data = torch.load(config['model_path'], map_location='cpu')
    logger.info('rank {}, model path: {}'.format(rank, config['model_path']))
    logger.info('rank {}, output dir: {}'.format(rank, config['out_dir']))
    logger.info('rank {}, model\'s epoch: {}'.format(rank, model_data['epoch']))

    config['experts']['stam']['pretrain'] = False
    stam = VideoExpert(config['experts']['stam'])
    stam.load_state_dict(model_data['stam'])
    stam.to(device)
    
    sbert = TextExpert(config['experts']['sbert'])
    sbert.load_state_dict(model_data['sbert'])
    sbert.to(device)
    
    if config['fusion']['name'] == 'fc':
        fusion = MLPFusion(
            config['experts']['stam']['dim']+config['experts']['sbert']['dim'], 
            config['fusion']['dim'])
    fusion.load_state_dict(model_data['fusion'])
    fusion.to(device)

    stam.eval()
    sbert.eval()
    fusion.eval()

    cnt = 0
    t = time.time()
    total_step = len(dataloader)
    for batch in dataloader:
        vdata, vmask, tdata, tmask, vids = batch
        vdata = vdata.to(device)
        vmask = vmask.to(device)
        tdata = tdata.to(device)
        tmask = tmask.to(device)

        try:
            with torch.no_grad():
                vembeddings = stam(vdata, vmask)['pooled_feature']
                tembeddings = sbert(tdata, tmask)['pooled_feature']
                embeddings = fusion(torch.cat([vembeddings, tembeddings], dim=1))
                embeddings = embeddings.detach().cpu().numpy()
        except:
            print(vdata.size())
            print(vmask.size())
            print(tdata.size())
            print(tmask.size())

        for i, vid in enumerate(vids):
            try:
                np.save(osp.join(config['out_dir'], f'{vid}.npy'), embeddings[i], allow_pickle=True)
            except:
                print(vids, vdata.size(), vmask.size(), embeddings.shape)
        
        cnt += 1
        if cnt % 1000 == 0:
            t = time.time() - t
            logger.info('rank {}, use {:.4f} min/k'.format(rank, t/60))
            logger.info('rank {}, etc {} min'.format(rank, (t/60)*((total_step-cnt)/1000)))
            t = time.time()

    logger.info('rank {}, done'.format(rank))


def parse_config(args):
    with open(args.config_file) as f:
        config = json.load(f)
    
    config['logdir'] = osp.join(config['logdir'], config['expname'])
    config['model_path'] = args.model_path
    config['out_dir'] = args.out_dir
    
    os.makedirs(args.out_dir, exist_ok=True)

    return config
        

class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('model_path')
    parser.add_argument('data_file')
    parser.add_argument('out_dir')
    parser.add_argument('--port', type=str, default='23333')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    config = parse_config(args)

    logger = utils.define_logger(config['logdir'], 'emb.log')

    # Split dataset to different devices.
    config['ngpus'] = torch.cuda.device_count()
    size_gpus = [config['train']['size_per_gpu']*3 for _ in range(config['ngpus'])]
    partitioner = DataPartitioner(args.data_file, np.array(size_gpus)/np.sum(size_gpus))

    # Running.
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(run, nprocs=config['ngpus'], args=(config, partitioner))
