import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.model import CBOW_Model, Skipgram_Model

def get_model_class(model_name: str):
  return CBOW_Model if model_name == 'cbow' else Skipgram_Model

def get_optimizer_class(name: str):
  return optim.Adam

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
  lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
  return LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)

def save_config(config: dict):
  config_path = os.path.join(config['run_dir'], 'config.yaml')
  with open(config_path, 'w') as w: yaml.dump(config, w)

def save_vocab(vocab, config: dict):
  torch.save(vocab, os.path.join(config['run_dir'], 'vocab.pt'))
