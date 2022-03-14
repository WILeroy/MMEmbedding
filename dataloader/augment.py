import glob
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image


class RandomStack(torch.nn.Module):
    """ Copy and stack k times along the x/y direction of the frames. """
    
    def __init__(self, low=2, up=3):
        super().__init__()

        self.low = low
        self.up = up

    def forward(self, frames):
        stack_num = random.randint(self.low, self.up)
        if frames.size()[2] > frames.size()[3]:
            frames = torch.tile(frames, (1, 1, 1, stack_num))
        else:
            frames = torch.tile(frames, (1, 1, stack_num, 1))
        return frames


class RandomBorder(torch.nn.Module):
    """ Add borders of random style to the top & bottom or left & right of the frames. """
    
    def __init__(self, styles=['single', 'image', 'blur_image'],
                       low=0.2, up=0.5, border_img_dir=None):
        super().__init__()
        
        self.styles = styles
        self.low = low
        self.up = up
        if 'image' in self.styles or 'blur_image' in self.styles:
            assert border_img_dir is not None
            self.addimage_list = glob.glob(os.path.join(border_img_dir, '*.jpg'))

    def get_single_border(self, h, w, n_frames):
        if random.random() < 0.8:
            color = random.randint(0, 255)
        else:
            color = torch.randint(low=0, high=255, size=(1, 3, 1, 1))
        single = torch.tile(torch.ones((1, 3, h, w)) * color, [n_frames, 1, 1, 1]).to(torch.uint8)
        return single, single

    def get_image_border(self, h, w, n_frames, addimg, blur=False):
        t_rate = 1.0 * h / w
        if (addimg.size()[0] * 1.0 / addimg.size()[1]) > t_rate:
            th = int(addimg.size()[2] * 1.0 * t_rate)
            timg1 = T.Resize((h, w))(addimg[:, :th, :])
            timg2 = T.Resize((h, w))(addimg[:, -th:, :])
        else:
            tw = int(addimg.size()[1] * 1.0 / t_rate)
            timg1 = T.Resize((h, w))(addimg[:, :, :tw])
            timg2 = T.Resize((h, w))(addimg[:, :, -tw:])
        
        if blur:
            timg1 = T.GaussianBlur((5, 5), (4, 4))(timg1)
            timg2 = T.GaussianBlur((5, 5), (4, 4))(timg2)
        
        return (torch.tile(timg1.unsqueeze(0), (n_frames, 1, 1, 1)), 
                torch.tile(timg2.unsqueeze(0), (n_frames, 1, 1, 1)))

    def forward(self, frames):
        b_rate = self.low + random.random() * (self.up - self.low)
        h = frames.size()[2]
        w = frames.size()[3]
        if w < h:
            bh = h
            bw = int(w * h * b_rate / ((2 - 2 * b_rate) * bh))
        else:
            bw = w
            bh = int(w * h * b_rate / ((2 - 2 * b_rate) * bw))
        
        style = random.choice(self.styles)
        assert style in ['single', 'image', 'blur_image']

        if style == 'single':
            bimg1, bimg2 = self.get_single_border(bh, bw, frames.shape[0])
        elif style == 'image':
            addimg = T.PILToTensor()(Image.open(random.choice(self.addimage_list)))
            bimg1, bimg2 = self.get_image_border(bh, bw, frames.shape[0], addimg)
        else:
            addimg = T.PILToTensor()(Image.open(random.choice(self.addimage_list)))
            bimg1, bimg2 = self.get_image_border(bh, bw, frames.shape[0], addimg, blur=True)

        if bh == h:
            frames = torch.cat([bimg1, frames, bimg2], dim=3)
        else:
            frames = torch.cat([bimg1, frames, bimg2], dim=2)
        
        return frames


class RandomVideoAug():
    """ A combination of multiple video augmentation methods. (For Self-Supervised) """
    
    def __init__(self, config):
        self.colorT = T.RandomChoice([
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.1),
            T.Grayscale(3),
            T.RandomPosterize(bits=3, p=0.9),
        ])

        self.spatialT = T.RandomChoice([
            T.RandomResizedCrop(size=(224, 224), scale=(0.5, 1)),
            T.Compose([
                T.RandomHorizontalFlip(p=0.9),    
                T.Resize((224, 224))
            ]),
            T.Compose([
                T.RandomPerspective(distortion_scale=0.35, p=0.9),    
                T.Resize((224, 224))
            ]),
            T.Compose([
                T.RandomRotation(degrees=(-30, 30)),    
                T.Resize((224, 224))
            ]),
            T.Compose([
                T.RandomApply([RandomBorder(border_img_dir=config['border_img_dir'])], p=0.9),
                T.Resize((224, 224))
            ]),
            T.Compose([
                T.RandomApply([RandomStack()], p=0.9),    
                T.Resize((224, 224))
            ])
        ], p=[1, 1, 1, 1, 1, 0.5])

        self.otherT = T.RandomChoice([
            T.RandomApply([
                T.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 3))
            ], p=0.3)
        ])

        self.trans = T.Compose([
            self.colorT,
            self.otherT,
            self.spatialT
        ])

    def __call__(self, frames):
        return self.trans(frames)


class VideoAug():
    """ Regular video augmentation methods. (For Supervised) """
    
    def __init__(self):
        self.colorT = T.RandomChoice([
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.1),
            T.Grayscale(3),
            T.RandomPosterize(bits=3, p=0.9),
        ])

        self.spatialT = T.RandomChoice([
            T.RandomResizedCrop(size=(224, 224), scale=(0.75, 1)),
            T.Compose([
                T.RandomHorizontalFlip(p=0.9),    
                T.Resize((224, 224))
            ]),
        ], p=[1, 1])

        self.trans = T.Compose([
            self.colorT,
            self.spatialT
        ])

    def __call__(self, frames):
        return self.trans(frames)


def tsn_sample(num_tokens, num_samples, training):
    """ num_tokens >= num_samples
    args:
        num_tokens: int, num of total tokens
        num_samples: int, num of sampled tokens
        training: bool
  
    returns:
        indexes: tensor, sampled indexes of frames
    """

    if num_samples == 1: return torch.tensor([0], dtype=torch.long)

    base = torch.floor(
        (num_tokens - 1) * torch.arange(0, num_samples).to(torch.float) / (num_samples - 1))

    if training:
        offset_range = base[1:] - base[:-1]
        base[:-1] += torch.rand((offset_range.size()[0],)) * offset_range
        indexes = torch.floor(base)
    else:
        indexes = base
    
    return indexes.to(torch.long)


def temporal_aug(total_num, max_length):
    """ temporal augmentation by shift window. """

    num_samples = min(total_num, max_length)
    
    if num_samples < total_num * 0.75:
        start = int(random.random() * 0.25 * total_num)
        return tsn_sample(int(total_num * 0.75), num_samples, True) + start
    else:
        return tsn_sample(total_num, num_samples, True)
    