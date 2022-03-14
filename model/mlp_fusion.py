import torch.nn as nn
import torch.nn.functional as F


class MLPFusion(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(MLPFusion, self).__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x
