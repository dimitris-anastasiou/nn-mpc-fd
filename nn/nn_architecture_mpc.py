# NN Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self, num_links, time_step):
		super().__init__()
		self.num_links = num_links
		self.time_step = time_step

	def forward(self, x):
		qddot = self.compute_qddot(x)
		state = x[:, :2*self.num_links]
		next_state = self.compute_next_state(state, qddot)
		return next_state

	def compute_next_state(self, state, qddot):

		# Extract current positions (q) and velocities (q_dot)
		q = state[:, :self.num_links]
		q_dot = state[:, self.num_links:2*self.num_links]

		# Compute next positions (q_next) and velocities (q_dot_next)
		q_next = q + q_dot * self.time_step
		q_dot_next = q_dot + qddot * self.time_step

		# Concatenate to form the next state vector
		next_state = torch.cat((q_next, q_dot_next), dim=1)
		return next_state

	def compute_qddot(self, x):
		pass

class Model2Link(Model):
	def __init__(self, time_step):
		super().__init__(2, time_step)
		# Define the neural network layers
		self.fc1 = nn.Linear(6, 128)     # Input layer (6 inputs -> 2 joint angles, 2 velocities, 2 actions)
		self.fc2 = nn.Linear(128, 128)   # Hidden layer
		self.fc3 = nn.Linear(128, 2)     # Output layer (2 outputs -> qddot for each joint)

	def compute_qddot(self, x):
		# Neural network forward pass to compute qddot
		x = F.relu(self.fc1(x))  	# Activation function for hidden layer
		x = F.relu(self.fc2(x))  	# Activation function for second hidden layer
		qddot = self.fc3(x)  		 	# Output layer, no activation function
		return qddot
