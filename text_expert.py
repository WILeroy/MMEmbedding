from transformers import BertTokenizer, BertModel
import torch.nn as nn


class TextExpert(nn.Module):
  def __init__(self, model_id, model_dir, max_length=20):
    """
    args:
      tokenizer:
      model:
      max_length: int
    """
    super().__init__()
    self.max_length = max_length

    self.tokenizer = BertTokenizer.from_pretrained(model_id,
                                                   cache_dir=model_dir)
    self.model = BertModel.from_pretrained(model_id,
                                           cache_dir=model_dir)

  def forward(self, raw_texts):
    """
    args:
      raw_texts: list, [raw_text1, raw_text2, ...]

    returns:
      batch_token: dict, {
        'input_ids': tensor [b, max_length],
        'token_type_ids': tensor [b, max_length],
        'attention_mask': tensor [b, max_length]
      }
      batch_feature: BaseModelOutputWithPoolingAndCrossAttentions, {
        'last_hidden_state': tensor [b, max_length, feature_dim],
        'pooler_output': tensor [b, feature_dim]
      }
    """
    assert isinstance(raw_texts, list)

    batch_token = self.tokenizer(
        raw_texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=self.max_length)

    batch_feature = self.model(**batch_token)
    
    return batch_token, batch_feature

#import time
#texter = TextExpert('bert-base-multilingual-cased', 'data/text_expert', max_length=20)
#t = time.time()
#tokens, features_text = texter(['我爱中国就是I love china', '今天天气真好啊今天天气真好啊今天天气真好啊今天天气真好啊今天天气真好啊今天天气真好啊'])
#print(time.time() - t)

#print(tokens['input_ids'])
#print(tokens['token_type_ids'])
#print(tokens['attention_mask'])
#print(features_text['last_hidden_state'])
#print(features_text['pooler_output'])