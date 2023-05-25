import os
import numpy as np
import torch
from torchinfo import summary
from tensorboardX import SummaryWriter

class Trainer:
  def __init__(self, config):
    self.C = config
    self.writer = SummaryWriter(config['run_dir'])

  def train(self):
    C = self.C
    for epoch in range(C['epochs']):
      self.writer.add_scalar('learning_rate', C['lr_scheduler'].get_last_lr(), epoch)
      self._train_epoch(epoch)
      self._validate_epoch(epoch)

      C['lr_scheduler'].step()
      if C['checkpoint_frequency']: self._save_checkpoint(epoch)
      print()

  def _train_epoch(self, epoch):
    C = self.C
    train_dataloader = C['train_dataloader']
    device, model, optimizer, criterion = C['device'], C['model'], C['optimizer'], C['criterion']
    steps = C['train']['steps']

    model.train()
    running_loss = []

    for i, (inputs, labels) in enumerate(train_dataloader, 1):
      if epoch == 0 and i == 1:
        summary(model, input_data=inputs, col_names=['input_size', 'output_size', 'num_params'])
        print()

      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      loss = criterion(model(inputs), labels)
      loss.backward()
      optimizer.step()  # モデルを更新する
      running_loss.append(loss.item())

      if i == steps: break

    self.writer.add_scalar('loss/train', np.mean(running_loss), epoch)

  def _validate_epoch(self, epoch):
    C = self.C
    valid_dataloader = C['valid_dataloader']
    device, model, criterion = C['device'], C['model'], C['criterion']
    steps = C['valid']['steps']

    model.eval()
    running_loss = []

    with torch.no_grad():
      for i, (inputs, labels) in enumerate(valid_dataloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        loss = criterion(model(inputs), labels)
        running_loss.append(loss.item())

        if i == steps: break

    self.writer.add_scalar('loss/valid', np.mean(running_loss), epoch)

  def _save_checkpoint(self, epoch):
    epoch_num = epoch + 1
    if epoch_num % self.C['checkpoint_frequency'] == 0:
      model_path = os.path.join(self.C['run_dir'], f'checkpoint_{epoch_num:03d}.pt')
      torch.save(self.C['model'], model_path)

  def save_model(self):
    model_path = os.path.join(self.C['run_dir'], 'model.pt')
    torch.save(self.C['model'], model_path)
