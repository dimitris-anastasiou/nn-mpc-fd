# NN Architecture

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import pickle
np.set_printoptions(suppress=True)


class DynamicDataset(Dataset):
    def __init__(self, data_file):
        data = pickle.load(open(data_file, "rb" ))
        # X: (N, 9), Y: (N, 6)
        self.X = data['X'].astype(np.float32)
        self.Y = data['Y'].astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    # ---
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # ---


def train(model):
    model.train()

    # ---
    total_loss = 0
    dt = 0.01

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        # Current
        current_positions = X[:, :3]
        current_velocities = X[:, 3:6]

        # Prepare inputs
        inputs = X.to(device)
        optimizer.zero_grad()

        # Predicted
        predicted_accelerations = model(inputs)
        predicted_next_velocities = current_velocities + predicted_accelerations * dt
        predicted_next_positions = current_positions + predicted_next_velocities * dt
        predicted_next_state = torch.cat((predicted_next_positions, predicted_next_velocities), dim=1)

        # Calculate loss with the actual next state
        loss = criterion(predicted_next_state, Y[:, :6])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Train Loss: {avg_loss:.4f}')
    return avg_loss
    # ---


def test(model):
    model.eval()

    # --
    total_loss = 0
    count = 0
    dt = 0.01

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)

            # Current
            current_positions = X[:, :3]
            current_velocities = X[:, 3:6]
            inputs = X

            # Predicted
            predicted_accelerations = model(inputs)
            predicted_next_velocities = current_velocities + predicted_accelerations * dt
            predicted_next_positions = current_positions + predicted_next_velocities * dt
            predicted_next_state = torch.cat((predicted_next_positions, predicted_next_velocities), dim=1)

            # Calculate loss between predicted next state and actual next state
            loss = criterion(predicted_next_state, Y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss
    # ---


# The ratio of the dataset used for testing
split = 0.2

# Do NOT change
# We are only using CPU, and GPU is not allowed.
device = torch.device("cpu")

dataset = DynamicDataset('data.pkl')
dataset_size = len(dataset)
test_size = int(np.floor(split * dataset_size))
train_size = dataset_size - test_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

# The name of the directory to save all the checkpoints
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
model_dir = os.path.join('models', timestr)

# Keep track of the checkpoint with the smallest test loss and save in model_path
model_path = None


# Define model, optimizer, criterion, and device setup
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
dt = 0.01

# Ensure the model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

best_test_loss = float('inf')


epochs = 50
for epoch in range(1, 1 + epochs):
    # ---
    print(f"Epoch {epoch}/{epochs}:")

    # Training
    train_loss = train(model)

    # Evaluation
    test_loss = test(model)

    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'
        model_path = os.path.join(model_dir, model_folder_name, 'dynamics.pth')
        if not os.path.exists(os.path.join(model_dir, model_folder_name)):
            os.makedirs(os.path.join(model_dir, model_folder_name))
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved to {model_path}")
        print("")
    else:
        print("Model not improved.")
        print("")

print("")
print(f'Best model saved at {model_path} with Test Loss: {best_test_loss:.4f}')
    # ---
