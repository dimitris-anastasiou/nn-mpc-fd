# MPC Controller Testing

import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
from render import Renderer
from score import *
import torch
import time
import math
np.set_printoptions(suppress=True)

# Teacher arm with 3 links
dynamics_teacher = ArmDynamicsTeacher(
    num_links=3,
    link_mass=0.1,
    link_length=1,
    joint_viscous_friction=0.1,
    dt=0.01)

arm = Robot(dynamics_teacher)
arm.reset()

gui = False

if gui:
  renderer = Renderer()
  time.sleep(1)

# Controller
controller = MPC()

# Resetting the arm will set its state so that it is in the vertical position,
# and set the action to be zeros
arm.reset()

# Choose the goal position you would like to see the performance of your controller
goal = np.zeros((2, 1))
goal[0, 0] = 2.5
goal[1, 0] = -0.7
arm.goal = goal

dt = 0.01
time_limit = 2.5
num_steps = round(time_limit/dt)
action = np.zeros((3, 1))

# Control loop
for s in range(num_steps):
  t = time.time()
  arm.advance()

  if gui:
    renderer.plot([(arm, "tab:blue")])
  time.sleep(max(0, dt - (time.time() - t)))

  if s % controller.control_horizon==0:
    state = arm.get_state()

    # Measuring distance and velocity of end effector
    pos_ee = dynamics_teacher.compute_fk(state)
    dist = np.linalg.norm(goal-pos_ee)
    vel_ee = np.linalg.norm(arm.dynamics.compute_vel_ee(state))
    print(f'At timestep {s}: Distance to goal: {dist}, Velocity of end effector: {vel_ee}')

    action = controller.compute_action(arm.dynamics, state, goal, action)
    arm.set_action(action)
