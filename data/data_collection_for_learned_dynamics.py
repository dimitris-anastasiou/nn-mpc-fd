# Data Collection

# Teacher arm with 3 links
dynamics_teacher = ArmDynamicsTeacher(
    num_links=2,
    link_mass=0.1,
    link_length=1,
    joint_viscous_friction=0.1,
    dt=0.01)

arm = Robot(dynamics_teacher)
arm.reset()

def collect_data(arm):

  # ---
  # Replace the X, and Y by your collected data
  # Control the arm to collect a dataset for training the forward dynamics.
  num_samples = 10000
  random_torque_SN = 10
  linearly_increasing_torque_SN = 6
  segmented_torque_SN = 6
  # Total number of simulations with num_samples steps each
  num_scenarios = random_torque_SN + linearly_increasing_torque_SN + segmented_torque_SN
  total_num_samples = num_scenarios * num_samples

  X = np.zeros((total_num_samples, arm.dynamics.get_state_dim() + arm.dynamics.get_action_dim()))
  Y = np.zeros((total_num_samples, arm.dynamics.get_state_dim()))

  gui = False

  # Initial arm state
  initial_state = np.zeros((arm.dynamics.get_state_dim(), 1))
  initial_state[0] = -math.pi / 2.0

  arm.set_state(initial_state)
  renderer = Renderer() if gui else None

  def apply_action_and_collect(arm, action, index):
    arm.set_action(action)
    current_state = arm.get_state().flatten()
    arm.advance()
    next_state = arm.get_state().flatten()

    X[index, :] = np.concatenate((current_state, action.flatten()))
    Y[index, :] = next_state

  current_index = 0
  for scenario in range(num_scenarios):
    arm.set_state(initial_state)
    for step in range(num_samples):
      if scenario < 50:  # Random torque scenario
        action = np.random.uniform(-1.5, 1.5, size=(arm.dynamics.get_action_dim(), 1))
      elif scenario < 80:  # Linearly increasing torque scenario
        start_torque = np.linspace(0.5, 1.5, 30)[scenario - 50]
        action = np.array([[start_torque * (step / num_samples)], [0]])
      else:  # Segmented torque scenario
        half_point = num_samples // 2
        if step < half_point:
          torque = np.linspace(-1, 1, 30)[scenario - 80]
        else:
          torque = np.linspace(-1, 1, 30)[-(scenario - 80)]
        action = np.array([[torque], [0]])

      apply_action_and_collect(arm, action, current_index)
      current_index += 1

      if gui:
        renderer.plot([(arm, 'tab:blue')])
  # ---

  return X, Y
