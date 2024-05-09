# Code by Alexandre Haddad-Delaveau
# w/ based off of Yann Bouteiller's "tmrl" project
# (which itself is based off of OpenAI Spinup code)
# https://github.com/yannbouteiller/tmrl


from math import floor, prod
from typing import Mapping, Sequence
import torch
import numpy as np
import torch.nn as nn

from util import collate

def calculate_output_dimensions(conv_layer, input_dimensions):
    h = input_dimensions[0]
    w = input_dimensions[1]

    h_out = floor((h + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)

    return (h_out,w_out)

def create_network(sizes, activation=nn.ReLU, output_activation=nn.ReLU):
    layers = []

    # Create layers
    for layer in range(len(sizes) - 1):
        activation_function = activation if layer < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[layer], sizes[layer+1]), activation_function()]
    
    # Return network
    return nn.Sequential(*layers)

class VanillaCNN(nn.Module):
    def __init__(self, q):
        super(VanillaCNN, self).__init__()
        self.q = q

        # Config
        self.output_shape = (64, 64)
        history_length = 5

        # Convolutional Layers
        self.conv1 = nn.Conv2d(history_length, 64, 8, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)

        # Calculate output dimensions
        h, w = calculate_output_dimensions(self.conv1, self.output_shape)
        h, w = calculate_output_dimensions(self.conv2, (h, w))
        h, w = calculate_output_dimensions(self.conv3, (h, w))
        self.h_out, self.w_out = calculate_output_dimensions(self.conv4, (h, w))
        self.output_channels = self.conv4.out_channels
        self.flat_features = self.output_channels * self.h_out * self.w_out
        
        # MLP
        self.mlp_input_features = self.flat_features + 3
        self.mlp_layers = [256, 256, 1] if self.q else [256, 256]
        self.mlp = create_network([self.mlp_input_features] + self.mlp_layers)
    
    def forward(self, x):
        if self.q:
            try:
                images, speed, gear, rpm, action = x
            except:
                images, speed, gear, rpm, action = x[0], x[1], x[2], x[3], x[4]
        else:
            try:
                images, speed, gear, rpm = x
            except:
                images, speed, gear, rpm = x[0], x[1], x[2], x[3]
        
        images = images.float()
            
        # Convolutional Layers
        x = torch.relu(self.conv1(images))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        # Count flat features
        x = x.view(-1, self.flat_features)
                
        # Concatenate with speed, gear, rpm, and actions
        if self.q:
            x = torch.cat([x, speed, gear, rpm], dim=-1)
        else:
            x = torch.cat([x, speed, gear, rpm], dim=-1)


        # MLP
        x = self.mlp(x)
        return x

class VanillaCNNActorModule(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Find action dimensions
        self.action_dimensions = action_space.shape[0]
        self.action_limit = action_space.high[0]

        # Create neural network and layers
        self.network = VanillaCNN(False)
        self.mean_layer = nn.Linear(256, self.action_dimensions)
        self.log_std_layer = nn.Linear(256, self.action_dimensions)
    
    def forward(self, observation):
        # Send through network
        output = self.network(observation)

        # Calculate mean and log-std 
        mean = self.mean_layer(output)
        log_std = self.log_std_layer(output)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Calculate normal distribution
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()

        # Calculate log probability of action
        probability = normal.log_prob(action).sum(axis=-1)
        probability -= (2*(np.log(2) - action - nn.functional.softplus(-2*action))).sum(axis=1)

        # Return action and probability
        return action, probability

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def act(self, observation):
        with torch.no_grad():
            action, _ = self.forward(observation)
            return action.squeeze().cpu().numpy()
        
    def act_(self, obs):
        obs = collate([obs], self.device)

        with torch.no_grad():
            action = self.act(obs)
        return action

    def to(self, device):
        self.device = device
        return super().to(device=device)

class VanillaCNNQFunction(nn.Module):
    def __init__(self):
        super().__init__()

        # Create network
        self.network = VanillaCNN(True)
    
    def forward(self, observation, action):
        # Send through network
        input = (*observation, action)
        return torch.squeeze(self.network(input), -1)

class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Create modules
        self.actor = VanillaCNNActorModule(observation_space, action_space)
        self.q1 = VanillaCNNQFunction()
        self.q2 = VanillaCNNQFunction()
    
    def act(self, observation):
        # Disable gradients
        with torch.no_grad():
            action, _ = self.actor(observation)
            return action.squeeze().cpu().numpy()