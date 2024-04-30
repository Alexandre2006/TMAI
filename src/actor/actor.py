# Code by Alexandre Haddad-Delaveau
from pathlib import Path
import torch

from gym.environment import GymEnvironment
from sac.actor import MLPSACActor
from buffer import shared_buffer, shared_weights
import datetime


class Actor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration (Spaces)
        self.environment = GymEnvironment()
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

        # Configuration (Actor)
        self.actor = MLPSACActor(self.observation_space, self.action_space).to(self.device)

        # Configuration (Episode Parameters)
        self.episode_max_length = 1000

        # Configuration (Backups)
        self.latest_path = Path("data/latest_actor.pkl")
        self.backup_path = Path("data/actor_backups")

        # Load Backup
        if self.backup_path.exists():
            print("Backup found. Loading actor...")
            self.load_backup()
        else: 
            print("No backup found. Starting fresh actor...")
    
    def update_weights(self):
        weights = shared_weights
        if weights != None:
            # Write new weights
            with open(self.latest_path, "wb") as file:
                file.write(weights)
            # Create backup
            self.current_model_count += 1
            time = datetime.datetime.now()
            path = self.backup_path / f"actor_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            with open(path, "wb") as file:
                file.write(weights)
            # Reload Actor (w/ new weights)
            self.actor = self.actor.load(self.latest_path, device=self.device)
        
    def act(self, observation):
        return self.actor.act_(observation)

    def reset(self):
        action = self.environment.unwrapped.default_action

        # Reset Environment
        new_observation, info = self.environment.reset()
        reward = 0.0
        terminated, truncated = False, False

        # Append sample to buffer
        sample = action, new_observation, reward, terminated, truncated, info
        shared_buffer.append(sample)

        # Return observation
        return new_observation, info

    def step(self, observation, truncate):
        # Act
        action = self.act(observation)

        # Step
        new_observation, reward, terminated, truncated, info = self.environment.step(action)

        # Check if truncation was requested
        if truncate and not terminated:
            truncated = True

        # Append sample to buffer
        sample = action, new_observation, reward, terminated, truncated, info
        shared_buffer.append(sample)

        # Return observation
        return new_observation, reward, terminated, truncated, info

    def run_episode(self):
        total_reward = 0.0
        steps = 0

        # Reset Environment
        observation, info = self.reset()

        # Run Episode
        for i in range(self.episode_max_length):
            observation, reward, terminated, truncated, info = self.step(observation, (i == self.episode_max_length - 1))
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        
        # Add statistics to buffer
        shared_buffer.train_return_stat += total_reward
        shared_buffer.train_steps_stat += steps
    
    def run_episodes(self, count): # Not used for training, just demos / testing
        for i in range(count):
            self.run_episode()
    
    def run(self, count):
        for i in range(count):
            # Run episode
            self.run_episode()

            # Update weights
            self.update_weights()