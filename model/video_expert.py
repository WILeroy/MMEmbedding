import torch
import torch.nn as nn
import torch.nn.functional as F

from .stam.transformer_model import STAM_224


class VideoExpert(nn.Module):
    def __init__(self, conf, reducer=None):
        super().__init__()

        self.num_classes = 0
        self.num_frames = conf['max_length']
        self.frame_size = conf['input_frame_size']
        self.pretrain = conf['pretrain']
        self.checkpoint_path = conf['pretrain_parms']

        self.stam = STAM_224(
            self.num_classes, self.frame_size, self.num_frames)
            
        if self.pretrain:
            state = torch.load(self.checkpoint_path, map_location='cpu')['model']
            del state['head.weight']
            del state['head.bias']
            self.stam.load_state_dict(state, strict=False)
    
        self.reducer = reducer

    def forward(self, data, mask):
        """
        args:
            data: tensor, [b, num_frames, 3, 224, 224]
            mask: tensor, [b, num_frames]
        """
        data = data.view(-1, 3, self.frame_size, self.frame_size)
        cls_emb, cls_logits, frame_embs = self.stam(data, mask)
    
        if self.reducer is not None:
            cls_emb = self.reducer(cls_emb)
            frame_embs = self.reducer(frame_embs)

        cls_emb_norm = F.normalize(cls_emb, p=2, dim=1)
        frame_embs = frame_embs.permute(1, 0, 2)
        frame_embs_norm = F.normalize(frame_embs, p=2, dim=2)

        outputs = {}
        outputs['pooled_feature'] = cls_emb_norm
        outputs['token_features'] = frame_embs_norm
        outputs['attention_mask'] = mask

        return outputs
