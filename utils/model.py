import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM

class CBOW_Model(nn.Module):
  def __init__(self, vocab_size: int):
    super(CBOW_Model, self).__init__()
    self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_DIMENSION, max_norm=EMBED_MAX_NORM)
    self.linear = nn.Linear(in_features=EMBED_DIMENSION, out_features=vocab_size)

  def forward(self, context):
    x = self.embeddings(context)
    x = x.mean(axis=1)  # 論文は sum を使っている
    return self.linear(x)

class Skipgram_Model(nn.Module):
  def __init__(self, vocab_size: int):
    super(Skipgram_Model, self).__init__()
    self.embeddings = nn.Embedding(num_embedding=vocab_size, embedding_dim=EMBED_DIMENSION, max_norm=EMBED_MAX_NORM)
    self.linear = nn.Linear(in_features=EMBED_DIMENSION, out_features=vocab_size)

  def forward(self, token):
    x = self.embeddings(token)
    return self.linear(x)
