import glob
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .augment import temporal_aug, tsn_sample


def video_loader(frames_path, max_length, training, video_transform, transform_cnt=1):
    """
    args:
        frames_path: str, frames dir
        max_length: int, max num of sampled frames
        training: bool
        video_transform: 
        transform_cnt: int, default 1, >= 1 if self-supervised 
  
    returns:
        videos: tensor, [transform_cnt, max_length, 3, 224, 224]
        masks: tensor, [transform_cnt, max_length], 0-padding
    """
    filelist = glob.glob(frames_path+'/*.jpg')
    filelist.sort()

    if len(filelist) == 0:
        return (torch.zeros((transform_cnt, max_length, 3, 224, 224), dtype=torch.float32), 
                torch.zeros((transform_cnt, max_length), dtype=torch.long))

    # load all frames
    frame_list = []
    for fname in filelist:
        try:
            img = T.PILToTensor()(Image.open(fname))
            frame_list.append(img)
        except:
            print(fname, 'error')
    frames = torch.stack(frame_list)

    # sample and augment
    videos = []
    if training:
        for i in range(transform_cnt):
            indexes_sampled = temporal_aug(frames.size()[0], max_length)
            videos.append(video_transform(frames[indexes_sampled]))
    else:
        indexes_sampled = tsn_sample(frames.size()[0], min(frames.size()[0], max_length), False)
        videos.append(T.Resize((224, 224))(frames[indexes_sampled]))

    #pad and stack
    masks = []
    for i in range(len(videos)):
        num_sampled = videos[i].size()[0]
        pad_size = (max_length-num_sampled, 3, 224, 224)
        if num_sampled < max_length:
            videos[i] = torch.cat([videos[i], torch.zeros(pad_size)], dim=0)
        mask_pad = torch.zeros((max_length, ), dtype=torch.long)
        mask_pad[:num_sampled] = 1
        masks.append(mask_pad)
  
    videos = torch.stack(videos) / 255.0
    masks = torch.stack(masks)
  
    return videos, masks


def audio_loader(melfeat_path, max_length, training, drop_rate=0, transform_cnt=1):
    """
    args:
        melfeat_path: str, path to audio mel-feature
        max_length: int, max num of sampled feature tokens
        training: bool
        drop_rate: float, default 0, rate to use empty audio
        transform_cnt: int, default 1, >= 1 if self-supervised 
  
    returns:
        data: tensor, [transform_cnt, max_length, 1, 96, 64]
        mask: tensor, [transform_cnt, max_length, ], 0-padding
    """
    if melfeat_path == 'empty':
        return (torch.zeros((transform_cnt, max_length, 1, 96, 64), dtype=torch.float32), 
                torch.zeros((transform_cnt, max_length), dtype=torch.float32))

    melfeat = torch.tensor(np.load(melfeat_path, allow_pickle=True))
    num_mel = melfeat.size()[0]
  
    audios = []
    if training:
        for i in range(transform_cnt):
            indexes_sampled = temporal_aug(num_mel, max_length)
            audios.append(melfeat[indexes_sampled])
    else:
        indexes_sampled = tsn_sample(num_mel, min(num_mel, max_length), False)
        audios.append(melfeat[indexes_sampled])

    masks = []
    for i in range(len(audios)):
        if random.random() < drop_rate:
            audios[i] = torch.zeros((max_length, 1, 96, 64), dtype=torch.float32)
            masks.append(torch.zeros((max_length, ), dtype=torch.long))
            continue
    
        audios[i] = torch.unsqueeze(audios[i], dim=1)
        num_sampled = audios[i].size()[0]
        pad_size = (max_length-num_sampled, 1, 96, 64)
        if num_sampled < max_length:
            audios[i] = torch.cat([audios[i], torch.zeros(pad_size, dtype=torch.float32)], dim=0)
        mask_pad = torch.zeros((max_length, ), dtype=torch.long)
        mask_pad[:num_sampled] = 1
        masks.append(mask_pad)
  
    audios = torch.stack(audios)
    masks = torch.stack(masks)

    return audios, masks


def text_loader(caption, tokenizer, max_length, training, drop_rate=0, transform_cnt=1):
    """
    args:
        caption: str
        tokenizer: str, transformers.AutoTokenizer
        max_length: int, max num of tokens
        training: bool
        drop_rate: float, default 0, rate to use empty caption
        transform_cnt: int, default 1, >= 1 if self-supervised 

    returns:
        tokens['input_ids']: tensor, [transform_cnt, max_length]
        tokens['attention_mask']: tensor, [transform_cnt, max_length], 0-padding
    """
    if training:
        if caption == '':
            text_batch = ['' for _ in range(transform_cnt)]
        else:
            text_batch = []
            for _ in range(transform_cnt):
                if random.random() < drop_rate:
                    text_batch.append('')
                else:
                    text_batch.append(caption)
    else:
        text_batch = [caption]

    tokens = tokenizer(text_batch,
                       return_tensors='pt',
                       padding='max_length',
                       truncation=True,
                       max_length=max_length)
    
    return tokens['input_ids'], tokens['attention_mask']
    