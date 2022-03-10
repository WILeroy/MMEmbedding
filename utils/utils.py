import logging
import os
import random

import numpy as np
import torch
  

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def convert_ddp_model_dict(src_state):
    dst_state = {}
    for k in src_state.keys():
        dst_state[k[7:]] = src_state[k]
    return dst_state


def define_logger(logdir, logname):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(logdir, logname))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(message)s'))
    logger.addHandler(handler)
    
    return logger
