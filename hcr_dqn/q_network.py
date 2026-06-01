# Human work: My personal implementation of the neural network used by vanilla DQN
# The network is intentionally a plain multilayer perceptron because our current
# state is already a compact numeric vector (from the env_wrapper that flattened the environment to a vector) 
# --> No need convolutional layers 
from __future__ import annotations 


# This code block is to attempts to import PyTorch 
try:
    import torch
    from torch import nn # Importing the neural network module from PyTorch --> Building blocks for constructing the Q-network architecture
except ModuleNotFoundError as exc:  
    # For convenience, I implement this exception to catch the error and print a more helpful message about installing PyTorch    
    raise ModuleNotFoundError(
        "No PyTorch :<. PyTorch is required for the DQN scaffold. Install torch in the "
        "project environment before training or evaluation."
    ) from exc


# Main Q-network class
# Vanilla DQN and my momentum-sensitive variation use the same network.
# This keeps the comparison focused on the reward-shaping idea, not on a bigger model.
class QNetwork(nn.Module):
    # Map a flat observation vector to one Q-value per action

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, int]):
        super().__init__()
        # Basic check to catch input/output dimension
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        
        # Build the MLP layers based on the hidden_sizes configuration
        # The network is a simple feedforward MLP with ReLU activations
        layers: list[nn.Module] = []
        previous_size = input_dim

        for hidden_size in hidden_sizes:
            # Each block is Linear --> ReLU
            # Read-ready since this is baseline
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU()) 
            previous_size = hidden_size

        # Final layer outputs one Q-value per action
        layers.append(nn.Linear(previous_size, output_dim))
        # Combine all the layers into a Sequential module for easy forward pass
        self.network = nn.Sequential(*layers)

    # Forward methods: runs the input through the network and returns the output Q-values
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run one forward pass through the MLP
        return self.network(x)
