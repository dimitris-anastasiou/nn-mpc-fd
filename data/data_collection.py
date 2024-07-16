# Data Collection

import numpy as np
import os
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import pickle
import math
from render import Renderer
import time
from custom_plot import plot_positions_velocities_with_fixed_bounds

# DO NOT CHANGE
# Teacher arm
dynamics_teacher = ArmDynamicsTeacher(
    num_links=3,
    link_mass=0.1,
    link_length=1,
    joint_viscous_friction=0.1,
    dt=0.01
)
arm_teacher = Robot(dynamics_teacher)

# ---
# X and Y should eventually be populated with your collected data
# Control the arm to collect a dataset for training the forward dynamics.
# Number of simulations for each scenario
random_torque_SN = 50
linearly_increasing_torque_SN = 30
segmented_torque_SN = 30
# Total number of simulations with 500 steps each
num_samples = (random_torque_SN+linearly_increasing_torque_SN+segmented_torque_SN) * 500
X = np.zeros((num_samples, arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim()))
Y = np.zeros((num_samples, arm_teacher.dynamics.get_state_dim()))

# We run the simulator for 5 seconds with a time step of 0.01 second,
# so there are 500 steps in total
num_steps = 500

# GUI visualization, this will drastically reudce the speed of the simulator!
# Set this to false once you understand how the code works
gui = False

# Define the initial state of the robot, such that it is vertical
initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position and velocity
initial_state[0] = -math.pi / 2.0

# Set the initial state of the arm. Input to set_state() should be of shape (6, 1)
arm_teacher.set_state(initial_state)

# Define the action, applying 1Nm torque to the first joint
action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
action[0] = 0

# Set the action. Input to set_action() should be of shape (3, 1)
arm_teacher.set_action(action)

# Initialize the GUI
if gui:
    renderer = Renderer()
    time.sleep(1)

def reset_simulation():
    arm_teacher.set_state(initial_state)

# Generate discrete torque values for each scenario
torques_random_1 = np.linspace(-1.5, -1.4, 10)
torques_random_2 = np.linspace(-1.4, 1.5, 40)
torques_random = np.concatenate([torques_random_1, torques_random_2])
torques_linear_start = np.linspace(0.5, 1.5, linearly_increasing_torque_SN)
torques_segmented_first_half = np.linspace(-1, 1, segmented_torque_SN)
torques_segmented_second_half = np.linspace(-1, 1, segmented_torque_SN)

def random_torque_scenario(simulation_index, step, num_steps):
    torque = torques_random[simulation_index]
    return np.array([[torque], [0], [0]])

def linearly_increasing_torque_scenario(simulation_index, step, num_steps):
    end_torque = torques_linear_start[simulation_index]
    return np.array([[(step / num_steps) * end_torque], [0], [0]])

def segmented_torque_scenario(simulation_index, step, num_steps):
    if step < num_steps / 2:
        torque = torques_segmented_first_half[simulation_index]
    else:
        torque = torques_segmented_second_half[simulation_index]
    return np.array([[torque], [0], [0]])

# Initialize indices for storing samples and scenario function
current_index = 0
scenarios = [
    (random_torque_scenario, random_torque_SN),
    (linearly_increasing_torque_scenario, linearly_increasing_torque_SN),
    (segmented_torque_scenario, segmented_torque_SN)
]

for scenario_function, num_runs in scenarios:
    for simulation_index in range(num_runs):
        reset_simulation()
        for s in range(num_steps):
            # Pass the current simulation index to the scenario function
            action = scenario_function(simulation_index, s, num_steps)
            state = arm_teacher.get_state().flatten()
            arm_teacher.set_action(action)
            arm_teacher.advance()
            new_state = arm_teacher.get_state().flatten()

            # Store data
            X[current_index, :] = np.concatenate((state, action.flatten()))
            Y[current_index, :] = new_state
            current_index += 1
# ---

# Save the collected data in the data.pkl file
data = {'X': X, 'Y': Y}
pickle.dump(data, open( "data.pkl", "wb" ) )

# Get the size of the file
file_size_mb = os.path.getsize('data.pkl')/1024/1024
print(f"File size: {file_size_mb:.2f} MB")
