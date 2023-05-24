import os
import numpy as np
import pandas as pd
import torch

dir = 'var/cbow_WikiText2'
device = torch.device('cpu')
model = torch.load(os.path.join(dir, 'model.pt'), map_location=device)
vocab = torch.load(os.path.join(dir, 'vocab.pt'))

embeddings, *_ = model.parameters()
embeddings = embeddings.cpu().detach().numpy()

# 埋め込みの正規化：正規化しておけば内積だけで類似度を計算できる。
norms = (embeddings ** 2).sum(axis=1) ** (1/2)
embeddings_normalized = embeddings / norms[:, None]   # [1] None でベクトルから行列を作れる

def get_top_similar(word: str, topN: int = 10):
  id = vocab[word]
  if id == 0:
    print('Out of vocabulary word')
    return

  wv = embeddings_normalized[id]
  dists = embeddings_normalized @ wv
  return [(vocab.lookup_token(id), dists[id])
          for id in np.argpartition(dists, -(topN + 1))[-(topN + 1):-1]]

print(pd.DataFrame(get_top_similar('peace'), columns=['word', 'similarity']).sort_values('similarity', ascending=False))

# [1] Numpy: Divide each row by a vector element, https://stackoverflow.com/a/19602209/15578861
