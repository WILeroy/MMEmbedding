import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextExpert(nn.Module):
    def __init__(self, conf, reducer=None):
        super().__init__()

        self.model_id = conf['model_id']
        self.num_tokens = conf['max_length']

        self.model = AutoModel.from_pretrained(self.model_id)

        self.reducer = reducer

    def forward(self, data, mask):
        """
        args:
            data: [b, max_length], tokens created by tokenizer
            mask: [b, max_length]
        """
        model_output = self.model(input_ids=data, attention_mask=mask)
        token_features = model_output[0] #First element of model_output contains all token embeddings

        if self.reducer is not None:
            token_features = self.reducer(token_features)

        outputs = collections.OrderedDict()
        outputs['pooled_feature'] = F.normalize(self.mean_pooling(token_features, mask), p=2, dim=1)
        outputs['token_features'] = F.normalize(token_features, p=2, dim=1)
        outputs['attention_mask'] = mask
    
        return outputs

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def logging(self, logger):
        logger.info('TextExpert model_id: {}'.format(self.model_id))
        logger.info('TextExpert num_tokens: {}'.format(self.num_tokens))
        