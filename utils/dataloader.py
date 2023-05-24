import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import CBOW_N_WORDS, SKIPGRAM_N_WORDS, MIN_WORD_FREQUENCY, MAX_SEQUENCE_LENGTH

# 未知語は元々のデータセットに含まれたタグなので変更できない。`wiki.train.tokens` を参照のこと。
UNKNOWN_TOKEN = '<unk>'

def get_english_tokenizer():
  return get_tokenizer('basic_english', language='en')


def get_data_iterator(ds_name, ds_type, data_dir):
  cls = WikiText103 if ds_name == 'WikiText103' else WikiText2
  return to_map_style_dataset(cls(root=data_dir, split=ds_type))

def build_vocab(data_iter, tokenizer):
  vocab = build_vocab_from_iterator(map(tokenizer, data_iter),
                                    specials=[UNKNOWN_TOKEN],
                                    min_freq=MIN_WORD_FREQUENCY)
  vocab.set_default_index(vocab[UNKNOWN_TOKEN])
  return vocab

def collate_cbow(batch, pipeline=None):
  N_WORDS = CBOW_N_WORDS

  inputs, outputs = [], []
  for text in batch:
    token_ids = pipeline(text)

    if len(token_ids) < N_WORDS * 2 + 1: continue

    if MAX_SEQUENCE_LENGTH: token_ids = token_ids[:MAX_SEQUENCE_LENGTH]

    for idx in range(len(token_ids) - N_WORDS * 2):
      context = token_ids[idx : (idx + N_WORDS * 2 + 1)]
      token_id = context.pop(N_WORDS)
      inputs.append(context)
      outputs.append(token_id)

  return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)

def collate_skipgram(batch, pipeline=None):
  N_WORDS = SKIPGRAM_N_WORDS

  inputs, outputs = [], []
  for text in batch:
    token_ids = pipeline(text)

    if len(token_ids) < N_WORDS * 2 + 1: continue

    if MAX_SEQUENCE_LENGTH: token_ids = token_ids[:MAX_SEQUENCE_LENGTH]

    for idx in range(len(token_ids) - N_WORDS * 2):
      context = token_ids[idx : (idx + N_WORDS * 2 + 1)]
      token_id = context.pop(N_WORDS)
      for context_token_id in context:
        inputs.append(token_id)
        outputs.append(context_token_id)

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)

def get_dataloader_and_vocab(model_name='', ds_name='', ds_type='', data_dir='', batch_size='', shuffle=True, vocab=None, **other):
  data_iter = get_data_iterator(ds_name, ds_type, data_dir)
  print(f'{model_name}/{ds_type}: #{len(data_iter)}, batch_size: {batch_size}')

  tokenizer = get_english_tokenizer()

  if not vocab: vocab = build_vocab(data_iter, tokenizer)

  pipeline = lambda x: vocab(tokenizer(x))
  collate_fn = collate_cbow if model_name == 'cbow' else collate_skipgram

  return (DataLoader(data_iter,
                     batch_size=batch_size,
                     shuffle=shuffle,
                     collate_fn=partial(collate_fn, pipeline=pipeline)),
          vocab if vocab else build_vocab(data_iter, tokenizer))
