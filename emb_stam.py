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
from dataloader.partition import MetaPartitioner

import torch.nn as nn
import torch.nn.functional as F
class ReduceDim(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x

def run(rank, config, partitioner):
    logger = utils.define_logger(config['logdir'], 'emb.log')
    logger.info("Rank {} entering the 'run' function".format(rank))
    
    # Init rank.
    if config['device'] == 'gpu':
        torch.cuda.set_device(rank) # !!! important !!!
        device = torch.device(rank)
    else:
        device = 'cpu'

    # Get splited dataset.
    partition = partitioner.use(rank)

    dataset = VTDatasetEmbedding(partition, config['experts']['stam'], config['experts']['sbert'])
    logger.info('rank {}, dataset size: {}'.format(rank, len(dataset)))
    
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
    stam = VideoExpert(config['experts']['stam'], reducer=ReduceDim(768, 512))
    stam.load_state_dict(model_data['stam'])
    stam.to(device)
    
    #sbert = TextExpert(config['experts']['sbert'])
    #sbert.load_state_dict(model_data['sbert'])
    #sbert.to(device)
    
    #if config['fusion']['name'] == 'fc':
    #    fusion = MLPFusion(
    #        config['experts']['stam']['dim']+config['experts']['sbert']['dim'], 
    #        config['fusion']['dim'])
    #fusion.load_state_dict(model_data['fusion'])
    #fusion.to(device)

    stam.eval()
    #sbert.eval()
    #fusion.eval()

    cnt = 0
    t = time.time()
    total_step = len(dataloader)
    for batch in dataloader:
        vdata, vmask, tdata, tmask, vids = batch
        vdata = vdata.to(device)
        vmask = vmask.to(device)
        #tdata = tdata.to(device)
        #tmask = tmask.to(device)

        try:
            with torch.no_grad():
                vembeddings = stam(vdata, vmask)['pooled_feature']
                #tembeddings = sbert(tdata, tmask)['pooled_feature']
                #embeddings = fusion(torch.cat([vembeddings, tembeddings], dim=1))
                embeddings = vembeddings.detach().cpu().numpy()
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
    config['device'] = args.device
    config['njobs'] = args.njobs
    
    os.makedirs(args.out_dir, exist_ok=True)

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('model_path')
    parser.add_argument('data_file')
    parser.add_argument('out_dir')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--njobs', type=int, default=1, help='number of processes')
    args = parser.parse_args()

    config = parse_config(args)

    logger = utils.define_logger(config['logdir'], 'emb.log')

    # Split dataset to different devices.
    config['njobs'] = torch.cuda.device_count() if config['device'] == 'gpu' else config['njobs']
    size_gpus = [1 for _ in range(config['njobs'])]
    partitioner = MetaPartitioner(args.data_file, np.array(size_gpus)/np.sum(size_gpus))

    # Running.
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(run, nprocs=config['njobs'], args=(config, partitioner))
