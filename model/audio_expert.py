import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggish.vggish import VGGish


class AudioExpert(nn.Module):
    def __init__(self, conf, reducer=None):
        super().__init__()

        self.num_frames = conf['max_length']
        self.pretrain = conf['pretrain']
        self.checkpoint_path = conf['pretrain_parms']

        self.vggish = VGGish(
            [self.checkpoint_path['vggish'], self.checkpoint_path['vggish_pca']],
            pretrained=self.pretrain,
            postprocess=False
        )

        self.reducer = reducer
    
    def forward(self, data, mask):
        """
        args:
            mel_features: tensor, [b, num_frames, 1, 96, 64]
            attention_mask: tensor, [b, num_frames]
        """
        b = data.size()[0]

        batch_token = data.view(b * self.num_frames, 1, 96, 64)
        token_features = self.vggish(batch_token)
        token_features = token_features.view(b, self.num_frames, 128)
        pooled_feature = self.mean_pooling(token_features, mask)

        if self.reducer is not None:
            token_features = self.reducer(token_features)
            pooled_feature = self.reducer(pooled_feature)

        outputs = {}
        outputs['pooled_feature'] = F.normalize(pooled_feature, p=2, dim=1)
        outputs['token_features'] = F.normalize(token_features, p=2, dim=2)
        outputs['attention_mask'] = mask

        return outputs
  
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
