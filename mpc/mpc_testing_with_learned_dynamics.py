# MPC Testing with Learned Dynamics Model

import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
from render import Renderer
from score import *
import torch
import time

# Teacher arm with 3 links
dynamics_teacher = ArmDynamicsTeacher(
    num_links=2,
    link_mass=0.1,
    link_length=1,
    joint_viscous_friction=0.1,
    dt=0.01)

arm = Robot(dynamics_teacher)
arm.reset()

gui = False
action = np.zeros((arm.dynamics.get_action_dim(), 1))
if gui:
  renderer = Renderer()
  time.sleep(1)

# Controller
controller = MPC()
dynamics_student = ArmDynamicsStudent(
    num_links=2,
    link_mass=0.1,
    link_length=1,
    joint_viscous_friction=0.1,
    dt=0.01)
device = torch.device('cpu')

# model_path should have the path to the best model that you have trained so far
# which you would like to use for testing the controller
model_path = model_path
dynamics_student.init_model(model_path, 2, 0.01, device)

# Control loop
action = np.zeros((arm.dynamics.get_action_dim(), 1))
goal = np.zeros((2, 1))
goal[0, 0] = 2.7
goal[1, 0] = 0.5
arm.goal = goal

dt = 0.01
time_limit = 2.5
num_steps = round(time_limit/dt)
for s in range(num_steps):
  t = time.time()
  arm.advance()

  if gui:
    renderer.plot([(arm, "tab:blue")])
  time.sleep(max(0, dt - (time.time() - t)))

  if s % controller.control_horizon==0:
    state = arm.get_state()
    action = controller.compute_action(dynamics_student, state, goal, action)
    arm.set_action(action)
