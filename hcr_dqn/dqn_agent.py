# Human work: My Vanilla DQN agent for the HCR project that is Human implemented
# Theoretical base that I use:
# - the online Q-network
# - the target Q-network
# - the optimizer
# - epsilon-greedy action selection
# - one gradient update step
from __future__ import annotations

from dataclasses import asdict # For converting the config dataclass to a dictionary when saving checkpoints
import pickle # For backward compatibility when loading older checkpoints that may contain untrusted types such as WindowsPath in the saved metadata
from pathlib import Path # For handling file paths in a way that works across different operating systems
import random # For epsilon-greedy action selection
import numpy as np # Numpy yay

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

# Avoid circular imports between dqn_agent and q_network
if __package__ in (None, ""):  
    from q_network import QNetwork
else:
    from .q_network import QNetwork


# Main DQN agent class - Vanilla DQN agent
def load_checkpoint_payload(path, device):
    # Load checkpoint data while staying compatible with both newer and older PyTorch defaults

    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        # Older local checkpoints may contain types such as WindowsPath in the
        # saved metadata. Those are trusted artifacts from this project, so we
        # fall back to full loading for backward compatibility.
        return torch.load(path, map_location=device, weights_only=False)


class DQNAgent:
    # The constructor initializes the online and target Q-networks, the optimizer, the loss function, and the epsilon value for exploration
    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # GPU is used if available, otherwise CPU is completely fine.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the online and target Q-networks taken from the q_network.py file
        # Both networks have the same architecture defined by the hidden_sizes parameter in the config
        self.online_network = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        # Initilize target newtork with same weights as online network, but eval mode since no training for target network
        self.target_network = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        # Copy weights from online network to target network
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # Use optimizer Adam (common for DQN)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate, # Learning rate from the config
        )
        # MSE loss for the Bellman update
        self.loss_fn = nn.MSELoss()

        self.epsilon = config.epsilon_start
        self.training_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        # Choose an action using epsilon-greedy exploration (explore=True) or purely greedy (explore=False)
        if explore and random.random() < self.epsilon: # With probability epsilon, select a random action for exploration
            return random.randrange(self.action_dim) 

        # If nto random, select action with highest Q-value 
        state_tensor = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )
        # Use torch.no_grad() to avoid tracking gradients during action selection (since no training)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        # return the index of the action with the highest Q-value, converting from a PyTorch tensor to a Python integer
        return int(torch.argmax(q_values, dim=1).item()) 

    def train_step(self, batch: dict[str, np.ndarray]) -> float:
        # Run one Bellman update and return the scalar loss value
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Gather the Q-values for the actions that were actually taken.
        current_q_values = self.online_network(states).gather(1, actions)

        with torch.no_grad():
            # Vanilla DQN target:
            # reward + gamma * max_a' Q_target(next_state, a')
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True).values
            target_q_values = rewards + (1.0 - dones) * self.config.gamma * next_q_values
        # MSE loss between current Q-values and target Q-values
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation phase: zero gradients, compute new gradients, and take an optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Increment training steps
        self.training_steps += 1
        # Return loss
        return float(loss.item())

    def update_target_network(self) -> None:
        # Copy the online network weights into the target network to have the latest learned Q-values
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self) -> None:
        # Move epsilon slowly toward its floor value
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def save(self, path) -> None:
        # Save enough information to resume or evaluate later

        config_dict = asdict(self.config)
        # Convert path objects in config to strings for better compatibility
        for key, value in list(config_dict.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value)
        # Save the state dicts of online and target networks, optimizer, epsilon, training step and config to checkpoint for evaluation/watch checkpoint
        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "config": config_dict,
            },
            path,
        )

    def load(self, path) -> None:
        # Load a saved checkpoint back into this agent

        checkpoint = load_checkpoint_payload(path, device=self.device)
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self.training_steps = int(checkpoint.get("training_steps", 0))


# My variation: Momentum-Sensitive DQN
# This keeps the normal DQN network and Bellman update, but changes the reward
# stored in replay memory so the agent is encouraged to move forward smoothly.
class MomentumSensitiveDQNAgent(DQNAgent):
    # Vanilla DQN with a small custom reward shaping rule for stable motion

    def __init__(self, state_dim: int, action_dim: int, config) -> None:
        # Reuse the full vanilla DQN setup first: networks, optimizer, epsilon, and target network.
        super().__init__(state_dim, action_dim, config)

        # These values are intentionally small because this is only a bonus/penalty on top of the environment reward
        self.momentum_bonus_scale = getattr(config, "momentum_bonus_scale", 0.05)
        self.stall_penalty = getattr(config, "momentum_stall_penalty", 0.02)
        self.oscillation_penalty = getattr(config, "momentum_oscillation_penalty", 0.02)
        self.forward_streak_required = getattr(config, "momentum_forward_streak_required", 3)
        self.stall_patience = getattr(config, "momentum_stall_patience", 30)
        self.progress_clip = getattr(config, "momentum_progress_clip", 5.0)
        self.angle_limit_deg = getattr(config, "momentum_angle_limit_deg", 45.0)
        self.angle_penalty_scale = getattr(config, "momentum_angle_penalty_scale", 0.001)
        self.air_penalty = getattr(config, "momentum_air_penalty", 0.01)
        self.back_wheel_penalty = getattr(config, "momentum_back_wheel_penalty", 0.005)

        # These variables track short-term behavior inside one episode.
        self.previous_score: float | None = None
        self.previous_action: int | None = None
        self.forward_streak = 0
        self.stall_steps = 0
        self.oscillation_steps = 0

    def reset_episode_reward_state(self) -> None:
        # Reset the custom motion tracking at the start of each episode
        self.previous_score = None
        self.previous_action = None
        self.forward_streak = 0
        self.stall_steps = 0
        self.oscillation_steps = 0

    def shape_reward(
        self,
        reward: float,
        action: int,
        info: dict | None = None,
        state: np.ndarray | None = None,
        next_state: np.ndarray | None = None,
    ) -> float:
        # Add a small motion-stability bonus or penalty before storing the transition

        if info is None:
            info = {}

        # The game score is based on the farthest forward distance reached by the car.
        current_score = float(info.get("score", 0.0))

        if self.previous_score is None:
            progress_delta = 0.0
        else:
            progress_delta = current_score - self.previous_score

        # The flattened observation order comes from env_wrappers.py:
        # 0=x position, 2=chassis angle, 5=back wheel on ground, 6=front wheel on ground.
        x_delta = 0.0
        chassis_angle = 0.0
        back_wheel_on_ground = 1.0
        front_wheel_on_ground = 1.0
        if state is not None and next_state is not None and len(next_state) >= 7:
            x_delta = float(next_state[0] - state[0])
            chassis_angle = float(next_state[2])
            back_wheel_on_ground = float(next_state[5])
            front_wheel_on_ground = float(next_state[6])

        # A forward step means the car actually moved forward or improved the game score.
        motion_progress = max(progress_delta, x_delta)
        is_forward_step = motion_progress > 0.0
        if is_forward_step:
            self.forward_streak += 1
            self.stall_steps = 0
        else:
            self.forward_streak = 0
            self.stall_steps += 1

        shaped_reward = float(reward)

        # Bonus: if the car keeps making forward progress for several steps, reward that smooth momentum
        if self.forward_streak >= self.forward_streak_required:
            clipped_progress = min(max(motion_progress, 0.0), self.progress_clip)
            shaped_reward += self.momentum_bonus_scale * clipped_progress

        # Penalty: large chassis angle means the car is flipping or balancing too aggressively
        angle_excess = max(0.0, abs(chassis_angle) - self.angle_limit_deg)
        shaped_reward -= self.angle_penalty_scale * min(angle_excess, 180.0)

        # Penalty: discourage fully airborne motion because it often comes from unstable flips
        if back_wheel_on_ground < 0.5 and front_wheel_on_ground < 0.5:
            shaped_reward -= self.air_penalty

        # Penalty: discourage long back-wheel-only balancing, but keep it small because hill climbing sometimes needs it
        if back_wheel_on_ground >= 0.5 and front_wheel_on_ground < 0.5:
            shaped_reward -= self.back_wheel_penalty

        # Penalty: if the car stops making useful progress for too long, discourage that behavior
        if self.stall_steps >= self.stall_patience:
            shaped_reward -= self.stall_penalty

        # Penalty: in the discrete action space, the two edge actions are opposite drive commands
        # Switching between them without progress is treated as unstable oscillation
        if self.previous_action is not None and {self.previous_action, int(action)} == {0, 2}:
            self.oscillation_steps += 1
        else:
            self.oscillation_steps = max(0, self.oscillation_steps - 1)

        if self.oscillation_steps > 0 and progress_delta <= 0.0:
            shaped_reward -= self.oscillation_penalty

        self.previous_score = current_score
        self.previous_action = int(action)
        return shaped_reward
