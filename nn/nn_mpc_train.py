# NN MPC Train

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import pickle
import torch.optim as optim
import argparse
import time

class DynamicDataset(Dataset):
  def __init__(self, datafile):
    data = pickle.load(open(datafile, 'rb'))
    # X: (N, 6), Y: (N, 4)
    self.X = data['X'].astype(np.float32)
    self.Y = data['Y'].astype(np.float32)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]


def train_one_epoch(model, train_loader, optimizer, criterion, device):
  model.train()
	# ---
  running_loss = 0.0
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)
	# ---
    return running_loss / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
  model.eval()
	# --
  test_loss = 0.0
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      test_loss += loss.item() * inputs.size(0)
  # --
  return test_loss / len(test_loader.dataset)

def train_forward_model():

  # Keep track of the checkpoint with the smallest test loss and save in model_path
  model_path = None
  max_test_loss = 1e4
  model = Model2Link(0.01)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.MSELoss()
  device = torch.device("cpu")

  datafile = 'dataset/data.pkl'
  split = 0.2
  dataset = DynamicDataset(datafile)
  dataset_size = len(dataset)
  test_size = int(np.floor(split * dataset_size))
  train_size = dataset_size - test_size
  train_set, test_set = random_split(dataset, [train_size, test_size])

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

  # The name of the directory to save all the checkpoints
  timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
  model_dir = os.path.join('models', timestr)

  epochs=50
  best_test_loss = float('inf')
  for epoch in tqdm.tqdm(range(1, 1 + epochs), desc='Training Epochs'):
    # --
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    test_loss = test(model, test_loader, criterion, device)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        model_path = os.path.join(model_dir, f'epoch_{epoch:04d}_loss_{test_loss:.8f}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, 'dynamics.pth'))
        print(f'model saved to {os.path.join(model_path, "dynamics.pth")}')
    # --

  return os.path.join(model_path, "dynamics.pth")
