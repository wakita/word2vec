import atexit
import argparse
from datetime import datetime
import yaml
import os
from subprocess import Popen
import torch
import torch.nn as nn
from torchinfo import summary

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import get_model_class, get_optimizer_class, get_lr_scheduler, save_config, save_vocab

def train(config):
  os.makedirs(config['run_dir'], exist_ok=True)

  train_dataloader, vocab = get_dataloader_and_vocab(ds_type='train', batch_size=config['train']['batch_size'],
                                                     **config)
  valid_dataloader, _     = get_dataloader_and_vocab(ds_type='valid', batch_size=config['valid']['batch_size'],
                                                     vocab=vocab, **config)

  vocab_size = len(vocab.get_stoi())
  print(f'Vocabulary size: {vocab_size}\n')

  model = (get_model_class(config['model_name']))(vocab_size=vocab_size)
  criterion = nn.CrossEntropyLoss()

  optimizer = (get_optimizer_class(config['optimizer']))(model.parameters(), lr=config['learning_rate'])
  lr_scheduler = get_lr_scheduler(optimizer, config['epochs'], verbose=True)

  #device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  device = torch.device('cpu')

  C = dict(config,
           **dict(model=model,
                  train_dataloader=train_dataloader, train_steps=config['train']['steps'],
                  valid_dataloader=valid_dataloader, valid_steps=config['valid']['steps'],
                  criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                  device=device))
  trainer = Trainer(C)

  trainer.train()
  print('\nTraining finished.')

  trainer.save_model()
  trainer.save_loss()
  save_vocab(vocab, C)
  save_config(config)
  print('\nModel artifacts saved to folder:', C['run_dir'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True, help='path to yaml config')
  args = parser.parse_args()

  with open(args.config) as f:
    C = yaml.safe_load(f)
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    C['run_dir'] = os.path.join('var', f'{C["model_name"]}_{C["dataset"]}', now)

    # Tensorboard サーバを起動。URL が表示されるので、それをブラウザで開けば学習の様子が見られる。
    p = Popen(['tensorboard', '--logdir', C['run_dir']])
    atexit.register(lambda: p.terminate())

    train(C)
