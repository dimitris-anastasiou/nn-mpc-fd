# NN MPC Predict

from arm_dynamics_base import ArmDynamicsBase

class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Initialize the model loading the saved model from provided model_path
        self.model = Model2Link(time_step).to(device)
        # Load the saved model from the provided model_path
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        # ---
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Use the loaded model to predict new state given the current state and action
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).view(1, -1)

            # Concatenate state and action to form the model input
            model_input = torch.cat((state_tensor, action_tensor), dim=1)

            # Predict the next state using the model
            with torch.no_grad():
                model_output = self.model(model_input)

            # Assuming model_output represents the change in state (i.e., delta state)
            new_state = state_tensor + model_output.squeeze(0)

            # Convert new_state back to a NumPy array and ensure it's returned in the correct shape
            return new_state.cpu().numpy().reshape(-1, 1)
            # ---
        else:
            return state
