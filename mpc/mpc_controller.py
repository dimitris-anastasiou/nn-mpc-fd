# MPC Controller

from collections import defaultdict
import numpy as np

class MPC:

  def __init__(self):
    # You can modify control_horizon
    self.control_horizon = 10
    # Define other parameters here
    self.base_step_size = 0.005              # Base step size
    self.step_size = 0.005                   # Step size for adjusting actions
    self.base_position_cost_weight = 2.07    # Weight for the cost component due to position
    self.base_velocity_cost_weight = 0.8     # Weight for the cost component due to velocity

  def adaptive_step_size(self, distance_to_goal, current_velocity):
      step_adjustment_factor = 1 + 0.5 * distance_to_goal + 0.5 * np.linalg.norm(current_velocity)
      return max(self.base_step_size / step_adjustment_factor, 0.001)

  def dynamic_gain_adjustment(self, distance_to_goal, current_velocity):
      position_gain = self.base_position_cost_weight * (2 / (1 + np.exp(-distance_to_goal)) - 1)
      velocity_gain = self.base_velocity_cost_weight * (1 + min(np.linalg.norm(current_velocity), 1))
      return position_gain, velocity_gain

  def compute_action(self, dynamics, state, goal, action):
    # You must return an array of shape (num_links, 1)
    num_links = dynamics.num_links
    best_action = np.zeros((num_links, 1))
    best_cost = np.inf
    for _ in range(self.control_horizon):
        next_state = dynamics.advance(state, action)
        pos_ee = dynamics.compute_fk(next_state)[:2, :]
        vel_ee = dynamics.compute_vel_ee(next_state)[:2, :]

        # Adjust step size and gain based on current state
        distance_to_goal = np.linalg.norm(pos_ee - goal)
        velocity = np.linalg.norm(vel_ee)
        self.step_size = self.adaptive_step_size(distance_to_goal, vel_ee)
        position_weight, velocity_weight = self.dynamic_gain_adjustment(distance_to_goal, vel_ee)

        # Calculate costs
        position_cost = np.sum((pos_ee - goal)**2)
        velocity_cost = np.sum(vel_ee**2)
        current_cost = position_weight * position_cost + velocity_weight * velocity_cost
        if current_cost < best_cost:
            best_cost = current_cost
            best_action = action.copy()

        # Adjust action based on deviation
        state_deviation = pos_ee - goal
        action_adjustment = np.zeros((num_links, 1))
        for i in range(num_links):
          action_adjustment[i, 0] = -state_deviation.flatten()[i % 2] * self.step_size
        action += action_adjustment

        # Update the current state for the next iteration
        state = next_state

    return best_action
