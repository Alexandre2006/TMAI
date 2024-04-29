# Code by Alexandre Haddad-Delaveau
# w/ based off of Yann Bouteiller's "tmrl" project
# (which itself is based off of OpenAI Spinup code)
# https://github.com/yannbouteiller/tmrl


from math import prod
from typing import Mapping, Sequence
import torch
import numpy as np
import torch.nn as nn

def create_network(sizes, activation, output_activation=nn.Identity):
    layers = []

    # Create layers
    for layer in range(len(sizes) - 1):
        activation_function = activation if layer < len(sizes) - 2 else output_activation
        layers += [nn.Lineaer(sizes[layer], sizes[layer+1]), activation_function()]
    
    # Return network
    return nn.Sequential(*layers)

class MLPSACActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        # Save spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # Find observation space dimensions
        try:
            observation_dimensions = sum(prod(size for size in space.shape) for space in observation_space)
            self.observation_is_tuple = True
        except:
            observation_dimensions = prod(observation_space.shape)
            self.observation_is_tuple = False
        
        # Find action space dimensions & range
        action_dimensions = action_space.shape[0]
        self.max_action = action_space.high[0]

        # Create Neural Network
        self.network = create_network([observation_dimensions] + list(hidden_sizes) + [action_dimensions], activation, activation)

        # Mean & Log-Std Layers
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dimensions)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dimensions)
    
    def forward(self, observation):
        # Prepare input
        if self.observation_is_tuple:
            observation = torch.cat(observation, -1)
        else:
            observation = torch.flatten(observation, start_dim=1)
        
        # Send through network
        output = self.network(observation)

        # Calculate mean, log-std, and std
        mean = self.mean_layer(output)
        log_std = self.log_std_layer(output)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Create normal distribution and get action
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()

        # Get log probability of action
        probability = normal.log_prob(action).sum(axis=-1)
        probability -= (2*(np.log(2) - action - nn.functional.softplus(-2*action))).sum(axis=1)

        # Tanh action
        action = torch.tanh(action)
        action = self.max_action * action

        # Return action
        return action, probability

    # Written by GitHub Copilot
    def collate(self, batch):
        first_elem = batch[0]

        if isinstance(first_elem, torch.Tensor):
            return torch.stack(batch, 0).to(self.device)

        elif isinstance(first_elem, np.ndarray):
            torch_batch = [torch.from_numpy(b) for b in batch]
            return self.collate(torch_batch)

        elif hasattr(first_elem, '__torch_tensor__'):
            torch_batch = [b.__torch_tensor__().to(self.device) for b in batch]
            return torch.stack(torch_batch, 0)
        
        elif isinstance(first_elem, Sequence):
            transposed_batch = list(zip(*batch))
            return type(first_elem)(self.collate(samples, self.device) for samples in transposed_batch)

        elif isinstance(first_elem, Mapping):
            return type(first_elem)((key, self.collate(tuple(d[key] for d in batch), self.device)) for key in first_elem)
        
        else:
            return torch.from_numpy(np.array(batch)).to(self.device)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def act(self, obs):
        # Don't need gradients because we aren't training here
        with torch.no_grad():
            action, _ = self.forward(obs)
            # Convert to numpy for easier use
            action = action.squeeze().cpu().numpy()
            # Convert to 1D array if 0D
            if not len(action.shape):
                action = np.expand_dims(action, 0)
            # Return the action
            return action

    def act_(self, obs):
        obs = self.collate([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs)
        return action

    def to(self, device):
        self.device = device
        return super().to(device=device)


class MLPQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # Find observation space dimensions
        try:
            observation_dimensions = sum(prod(size for size in space.shape) for space in observation_space)
            self.observation_is_tuple = True
        except:
            observation_dimensions = prod(observation_space.shape)
            self.observation_is_tuple = False

        # Find action space dimensions
        action_dimensions = action_space.shape[0]

        # Create Neural Network
        self.network = create_network([observation_dimensions + action_dimensions] + list(hidden_sizes) + [1], activation)

    def forward(self, observation, action):
        # Prepare input
        if self.observation_is_tuple:
            observation = torch.flatten((*observation, action), -1)
        else:
            observation = torch.cat((torch.flatten(observation, start_dim=1), action), -1)

        # Send through network
        network = self.network(observation)

        # Return Q-Value
        return torch.squeeze(network, -1)

class MLPSAC(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()

        # Get action limits
        self.max_action = action_space.high[0]

        # Construct Policy
        self.actor = MLPSACActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
    
    def act(self, observation):
        # Disable gradients (not training here)
        with torch.no_grad():
            # Get action
            action = self.actor(observation)

            # Convert to numpy
            action = action.squeeze().cpu().numpy()

            # Convert to 1D array if 0D
            if not len(action.shape):
                action = np.expand_dims(action, 0)
            
            # Return action
            return action