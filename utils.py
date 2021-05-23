import numpy as np

def choose_or_pad_to_len(features,
                         features_t,
                         max_tokens,
                         training,
                         shuffle=False,
                         seed=0):
  """Outputs a fixed length sequence of features from a variable length input.

  Performs a selection if there are too many input features.
  Pads the sequence if there are too few features.

  Args:
    features: Input features.
    features_t: Input features timestamps.
    max_tokens: Length of the output sequence.
    training: If True, the features will be deterministically sampled.
    shuffle: If True, the features are shuffled.
    seed: Seed used for the random shuffling.

  Returns:
    Fixed length sequence of features.
  """
  feature_dim = features.shape[-1]
  tensor = np.zeros((max_tokens, feature_dim))
  tensor_t = np.ones((max_tokens))
  tensor_ind = np.zeros((max_tokens))
  keep = min(len(features), max_tokens)
  if training:
    # If training, we randomly pick features
    pick = np.random.choice(len(features), size=keep, replace=False)
  else:
    # If not training, the choice of features is deterministic
    rng = np.random.RandomState(0)
    pick = rng.choice(len(features), size=keep, replace=False)
  pick = np.sort(pick)
  tensor[:keep, :] = features[pick]
  if shuffle and training:
    # Shuffle temporal encoding so that the model cannot use temporal
    # information.
    rng = np.random.RandomState(seed)
    tensor_t[:keep] = rng.shuffle(features_t[pick])
  else:
    tensor_t[:keep] = features_t[pick]
  tensor_ind[:keep] = 1
  return tensor, tensor_t, tensor_ind