# Neural Network-Based Forward Dynamics and Model Predictive Control for a n-Link Robotic Arm

## Objective
This project involves the implementation of advanced control systems for a 3-link robotic arm, combining neural network techniques with Model Predictive Control (MPC). The project is divided into several key phases: learning forward dynamics using neural networks, developing an MPC controller, and evaluating MPC controller using the learned dynamics model.

## Project Structure
- **data/**: Contains data collection code.
  - **data_collection.py**: Script for collecting data from the robotic arm.
  - **data_collection_for_mpc.py**: Script for collecting data from the robotic arm.
- **nn/**: Contains Python files for the neural network models.
  - **nn_architecture_train.py**: Script for training the neural network architecture.
  - **nn_predict.py**: Script for making predictions using the trained neural network.
  - **nn_architecture_for_mpc.py**: Neural network architecture adapted for MPC.
  - **nn_predict_for_mpc.py**: Script for making predictions using the MPC-adapted neural network.
  - **nn_train_for_mpc.py**: Script for training the neural network for MPC.
- **mpc/**: Contains Python files for the mpc controller.
  - **mpc_controller.py**: Implementation of the Model Predictive Control (MPC) class.
 
## Setup Instructions

1. Clone the repository
  ```sh
  git clone https://github.com/dimitris-anastasiou/nn-mpc-fd.git
  ```
2. Navigate to the project directory:
  ```sh
  cd nn-mpc-fd
