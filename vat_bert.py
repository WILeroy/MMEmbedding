import collections
import types
import torch

import torch.nn as nn

from base import BaseModel, ReduceDim
from bert import BertModel as MMT
from text_expert import TextExpert


expert_config = {
  'video': {
    'dim': 4,
    'id': 1,
    'num_tokens': 2
  },
  'audio': {
    'dim': 4,
    'id': 2,
    'num_tokens': 3
  },
  'text': {
    'dim': 4,
    'id': 3,
    'num_tokens': 3
  }
}


class VATBert(BaseModel):
  def __init__(self,
               expert_config,
               text_expert_id,
               text_expert_dir,
               device=None,
               text_max_length=20,
               same_dim=4,
               mmt_params=None
  ):
    super().__init__()

    self.modalities = list(expert_config.keys())
    self.same_dim = same_dim
    self.device = device

    self.video_expert = None
    self.audio_expert = None
    self.text_expert = TextExpert(
        text_expert_id, text_expert_dir, text_max_length)

    # Reducing dim of different expert output.
    self.dim_reduce = nn.ModuleDict()
    for mod in self.modalities:
      in_dim = expert_config[mod]['dim']
      self.dim_reduce[mod] = ReduceDim(in_dim, same_dim)
    
    #mmt_config = types.SimpleNamespace(**mmt_params)
    #self.mmt = MMT(mmt_config)

  def convert_features(self, features, mod, last_token_idx):
    b = features['pooled_feature'].size()[0] # batch size

    features_list = []       # expert output
    input_ids_list = []      # 0=[CLS], 1=[AGG], 2=[FEA]
    token_type_ids_list = [] # 0=[CLS], others=expert id
    position_ids_list = []   # 0=no position, 1=unknow, others=valid position
    attention_mask_list = [] # valid token or not

    # [AGG] token
    token_idx = last_token_idx + 1
    agg_token_idx = token_idx

    token_id = expert_config[mod]['id']

    features_list.append(features['pooled_feature'])
    input_ids_list.append(torch.full((b,), 1, dtype=torch.long))
    token_type_ids_list.append(torch.full((b,), token_id, dtype=torch.long))
    position_ids_list.append(torch.full((b,), 0, dtype=torch.long))
    attention_mask_list.append(torch.full((b,), 1, dtype=torch.long))

    # [FEA] token
    for idx in range(expert_config[mod]['num_tokens']):
      token_idx += 1
      features_list.append(features['token_features'][:, idx, :])
      input_ids_list.append(torch.full((b,), 2, dtype=torch.long))
      token_type_ids_list.append(torch.full((b,), token_id, dtype=torch.long))
      position_ids_list.append(features['token_timestamps'][:, idx])
      attention_mask_list.append(features['token_masks'][:, idx])

    return (features_list,
            input_ids_list, 
            token_type_ids_list,
            position_ids_list,
            attention_mask_list,
            agg_token_idx,
            token_idx)

  def build_tokens(self, video_features, audio_features, text_features):
    b = video_features['pooled_feature'].size()[0] # batch size
    
    features_list = []       # expert output
    input_ids_list = []      # 0=[CLS], 1=[AGG], 2=[FEA]
    token_type_ids_list = [] # 0=[CLS], others=expert id
    position_ids_list = []   # 0=no position, 1=unknow, others=valid position
    attention_mask_list = [] # valid token or not

    agg_token_idx_of = collections.OrderedDict()

    # [CLS] token
    token_idx = 0
    features_list.append(
        torch.full((b, self.same_dim), 0, dtype=torch.float).to(self.device))
    input_ids_list.append(torch.full((b,), 0, dtype=torch.long))
    token_type_ids_list.append(torch.full((b,), 0, dtype=torch.long))
    position_ids_list.append(torch.full((b,), 0, dtype=torch.long))
    attention_mask_list.append(torch.full((b,), 1, dtype=torch.long))

    #stack [AGG] token and [FEA] tokens.
    for features, mod in zip(
        [video_features, audio_features, text_features],
        ['video', 'audio', 'text']
    ):
      outputs = self.convert_features(features, mod, token_idx)
      features_list += outputs[0]
      input_ids_list += outputs[1]
      token_type_ids_list += outputs[2]
      position_ids_list += outputs[3]
      attention_mask_list += outputs[4]
      agg_token_idx_of[mod] = outputs[5]
      token_idx = outputs[6]

    features = torch.stack(features_list, dim=1).to(self.device)
    input_ids = torch.stack(input_ids_list, dim=1).to(self.device)
    token_type_ids = torch.stack(token_type_ids_list, dim=1).to(self.device)
    position_ids = torch.stack(position_ids_list, dim=1).to(self.device)
    attention_mask = torch.stack(attention_mask_list, dim=1).to(self.device)

    return {
      'features': features,
      'input_ids': input_ids,
      'token_type_ids': token_type_ids,
      'position_ids': position_ids,
      'attention_mask': attention_mask,
      'agg_token_idx_of': agg_token_idx_of
    }

  def forward(self,
              video,
              audio,
              text,         # list, ['text1', 'text2', ...]
              out='sim',    # 'sim' or 'emb'
              device=None):    
    pass

def build_tokens_unit_test():
  test_expert_config = {
    'video': {
      'dim': 4,
      'id': 1,
      'num_tokens': 2
    },
    'audio': {
      'dim': 4,
      'id': 2,
      'num_tokens': 3
    },
    'text': {
      'dim': 4,
      'id': 3,
      'num_tokens': 3
    }
  }

  video = {
    'pooled_feature': torch.tensor([[1, 2, 1, 1], [1, 3, 1, 3]]),
    'token_features': torch.tensor([
        [[2, 3, 2, 3], [2, 4, 2, 4]],
        [[1, 5, 5, 3], [2, 1, 3, 4]]
    ]),
    'token_timestamps': torch.tensor([[2, 3], [2, 4]]),
    'token_masks': torch.tensor([[1, 1], [1, 1]])
  }

  audio = {
    'pooled_feature': torch.tensor([[4, 5, 6, 7], [1, 2, 3, 4]]),
    'token_features': torch.tensor([
        [[6, 3, 6, 3], [6, 4, 6, 4], [0, 0, 0, 0]],
        [[6, 5, 5, 3], [6, 1, 3, 4], [0, 0, 0, 0]]
    ]),
    'token_timestamps': torch.tensor([[2, 3, 0], [2, 3, 0]]),
    'token_masks': torch.tensor([[1, 1, 0], [1, 1, 0]])
  }

  text = {
    'pooled_feature': torch.tensor([[10, 5, 6, 7], [10, 2, 3, 4]]),
    'token_features': torch.tensor([
        [[10, 3, 6, 3], [10, 4, 6, 4], [10, 5, 6, 5]],
        [[10, 5, 5, 3], [10, 1, 3, 4], [0, 0, 0, 0]]
    ]),
    'token_timestamps': torch.tensor([[2, 3, 5], [2, 4, 0]]),
    'token_masks': torch.tensor([[1, 1, 1], [1, 1, 0]])
  }

  vat = VATBert(test_expert_config, 'bert-base-multilingual-cased', 'data/text_expert')
  print(vat.build_tokens(video, audio, text))

if __name__ == '__main__':
  build_tokens_unit_test()