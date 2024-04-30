# Code by Alexandre Haddad-Delaveau
# w/ based off of Yann Bouteiller's "tmrl" project
# (which itself is based off of OpenAI Spinup code)
# https://github.com/yannbouteiller/tmrl

# Default values from: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/sac.py 
 
from copy import deepcopy
import itertools

import torch
from sac.actor import MLPSAC
from torch.optim import Adam
import numpy as np

def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


class SACTrainingAgent:
    def __init__(self, observation_space, action_space, device):
        # Configure Torch Device
        self.device = device

        # Configure Spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # Configure model
        self.model = MLPSAC(observation_space, action_space).to(self.device)
        self.model_target = disable_gradients(deepcopy(self.model))

        # Network Parameters
        self.gamma = 0.99
        self.polyak = 0.995
        self.alpha = 0.2
        self.learning_rate = 1e-3

        # Configure Optimizers
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), lr=self.learning_rate)

        # Configure target entropy
        self.target_entropy = -np.prod(action_space.shape)

        # Configure Entropy Learning
        # Did some research on this, it's pretty cool!
        # Basically, entropy is randomness. Do you want the actor to explore? High entropy. Repeatable actions? Low entropy.
        # The alpha parameter is a trade-off between exploration and repeatable actions.
        # So, to avoid this trade-off, we make entropy a learnable parameter.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.learning_rate)

    def get_actor(self):
        return self.model.actor

    def train(self, batch):
        # Unbatch the batch
        observations, actions, rewards, next_observations, dones, _ = batch

        # Get actor output
        action, probability = self.model.actor(observations)

        # Entropy Learning
        alpha = torch.exp(self.log_alpha.detach())
        loss_alpha = -(self.log_alpha * (probability + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()

        # Compute Q-Values
        q1 = self.model.q1(observations, actions)
        q2 = self.model.q2(observations, actions)

        # Compute Target Q-Values
        with torch.no_grad():
            # Find next actions and probabilities
            next_actions, next_probabilities = self.model.actor(next_observations)
            
            # Find next Q-Values
            next_q1 = self.model_target.q1(next_observations, next_actions)
            next_q2 = self.model_target.q2(next_observations, next_actions)

            # Find minimum Q-Value (target)
            next_q = torch.min(next_q1, next_q2)

            # Compute Target Q-Value
            target_q = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_probabilities)

        # Compute Critic Loss
        loss_q1 = ((q1 - target_q) ** 2).mean()
        loss_q2 = ((q2 - target_q) ** 2).mean()
        loss_critic = (loss_q1 + loss_q2) / 2

        # Update Critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Disable gradients for q1 and q2 (should cut down on computation)
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Compute Actor Loss
        q1 = self.model.q1(observations, action)
        q2 = self.model.q2(observations, action)
        q = torch.min(q1, q2)

        loss_actor = (alpha * probability - q).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Enable gradients for q1 and q2
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Update Target Networks
        with torch.no_grad():
            for target, source in zip(self.model_target.parameters(), self.model.parameters()):
                target.data.mul_(self.polyak)
                target.data.add_((1 - self.polyak) * source.data)
            
            return dict(
                loss_actor=loss_actor.item(),
                loss_critic=loss_critic.item(),
                loss_alpha=loss_alpha.item(),
                alpha=alpha.item()
            )