## NN Predict

from arm_dynamics_base import ArmDynamicsBase

class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Initialize the model loading the saved model from provided model_path
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        # ---
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Preparing the input tensor from state and action arrays
            state_action = np.concatenate((state.flatten(), action.flatten()), axis=0)
            state_action_tensor = torch.tensor(state_action, dtype=torch.float32).to(self.device).unsqueeze(0)

            # Model prediction
            with torch.no_grad():
                acceleration = self.model(state_action_tensor).cpu().numpy().flatten()

            # Euler integration for the next state
            new_velocity = state[3:] + acceleration.reshape((3, 1)) * dt
            new_position = state[:3] + new_velocity * dt
            new_state = np.concatenate([new_position, new_velocity])

            return new_state.reshape((6, 1))
            # ---
        else:
            return state.reshape((6, 1))
