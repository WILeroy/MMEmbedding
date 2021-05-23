"""Models base class."""
import abc

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
  """Base class for all models."""

  @abc.abstractmethod
  def forward(self, *inputs):
    """Forward pass logic."""
    raise NotImplementedError

  def __str__(self):
    """Model prints with number of trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return super().__str__() + f"\nTrainable parameters: {params}"


class ReduceDim(nn.Module):
  def __init__(self, input_dimension, output_dimension):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dimension, output_dimension)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x