from pathlib import Path
import time
from typing import Mapping, Sequence
import torch
import os
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from buffer import shared_buffer, shared_weights
from gym.interface import TrackmaniaInterface
from sac.agent import SACTrainingAgent
from util import collate

def save_trainer(trainer, backup_path): # Using temporary path to avoid corruption
    # Create temporary path from backup path
    temp_path = backup_path.with_suffix(".tmp")

    # Save to temporary path
    with open(temp_path, "wb") as file:
        pickle.dump(trainer, file)
    
    # Replace backup path with temporary path
    os.replace(temp_path, backup_path)

def load_trainer(backup_path):
    # Load trainer from backup path
    with open(backup_path, "rb") as file:
        trainer = pickle.load(file)
    
    return trainer

def run_trainer():
        # Set trainer backup path
        backup_path = Path("data/backup_trainer.pkl")

        # Attempt to load previous backup
        if os.path.exists(backup_path):
            print("Backup found. Loading trainer...")
            trainer = load_trainer(backup_path)
        # If no backup exists, create a new trainer
        else:
            print("No backup found. Creating new trainer...")
            trainer = Trainer()
            # Save trainer
            save_trainer(trainer, backup_path)
        
        while trainer.epoch < trainer.epochs:
            # Run epoch
            epoch_stats = trainer.run_epoch()

            # Log epoch statistics
            print("Epoch Statistics:")
            print(epoch_stats)

            # Backup trainer
            save_trainer(trainer, backup_path)

class Trainer:
    def __init__(self):

        # TODO: Figure out what type of device AMD Radeon Compute is
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration (Training Duration)
        self.steps = 100 # Steps per epoch
        self.rounds = 10 # Rounds per epoch - Statistics are updated each round
        self.epochs = 10 # Total epochs - Model is updated each epoch

        # Configuration (Training Parameters)
        self.model_update_interval = 10 # How often the model is updated
        self.max_training_steps_per_env_step = 1.0 # ALWAYS 1:1 for trackmania
        self.start_training = 0

        # Configuration (Buffer)
        self.update_buffer_interval = 10
        self.sleep_betweeen_buffer_retrieval_attempts = 1.0

        # Configuration (Spaces)
        interface = TrackmaniaInterface()
        self.action_space = interface.get_action_space()
        self.observation_space = interface.get_observation_space()

        # Configuration (Memory)
        self.memory = TrainerMemory(self.device, self.steps)

        # Configuration (Agent)
        self.agent = SACTrainingAgent(self.observation_space, self.action_space, self.device)

        # Keep track of total model updates
        self.total_updates = 0

        # Keep track of current epoch
        self.epoch = 0

        # Print start statistics
        self.total_samples = len(self.memory)
        print(f"Starting training with {self.total_samples} samples")
    
    def update_buffer(self):
        # Get shared buffer
        buffer = shared_buffer
        count = len(buffer.memory)

        # Append shared buffer to memory
        self.memory.append(shared_buffer)

        # Clear shared buffer
        shared_buffer.clear_memory()

        # Update statistics
        self.total_samples = len(self.memory.data)
        print(f"Memory updated with {count} samples, total samples: {self.total_samples}")
    
    def check_step_ratio(self):
        # Check if number of steps in buffer is equal to the number of steps in the environment
        # And that we have met the minimum required for training
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 else -1.0
        minimum_met = self.total_samples >= self.memory.minimum_samples

        # If we do not have enough samples for training, or the ratio is too high, wait for more samples
        if not minimum_met or (ratio > self.max_training_steps_per_env_step or ratio == -1.0):
            print("Not enough samples for training, or ratio too high. Waiting for more samples...")
            while not minimum_met or (ratio > self.max_training_steps_per_env_step or ratio == -1.0):
                # Update buffer
                self.update_buffer()

                # Check ratio and minimum
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 else -1.0
                minimum_met = self.total_samples >= self.memory.minimum_samples

                # Sleep for a bit if still not enough samples
                if not minimum_met or (ratio > self.max_training_steps_per_env_step or ratio == -1.0):
                    time.sleep(self.sleep_betweeen_buffer_retrieval_attempts)
            
            print("Enough samples for training. Continuing...")
    
    def run_epoch(self):
        stats_epoch = []

        # Repeat each round
        for round in range(self.rounds):
            # Log current round
            print(f"[{self.epoch}/{self.epochs} epochs][{round}/{self.rounds} rounds] - {round/self.rounds*100}% Complete")

            # Create stats for this round
            stats_round = []

            # Check ratio (measure time taken)
            check_ratio_start = time.time()
            self.check_step_ratio()
            check_ratio_end = time.time()

            # Measure sample time
            sample_previous = check_ratio_end

            for batch in self.memory:
                print(len(batch[0]))

                # Measure sample time
                sample_start = time.time()

                # Check if we need to update the buffer
                if self.total_updates % self.update_buffer_interval == 0:
                    self.update_buffer()
                
                buffer_end = time.time()

                # Train
                stats = self.agent.train(batch)

                train_end = time.time()

                # Update statistics
                stats["return_test"] = self.memory.test_return_stat
                stats["return_train"] = self.memory.train_return_stat
                stats["episode_length_test"] = self.memory.test_steps_stat
                stats["episode_length_train"] = self.memory.train_steps_stat
                stats["sampling_duration"] = sample_start - sample_previous
                stats["training_step_duration"] = train_end - sample_start
                stats_round += stats
                self.total_updates += 1

                # Check if we need to update the model
                if self.total_updates % self.model_update_interval == 0:
                    self.agent.get_actor().save("trainer.weights")
                    with open("trainer.weights", "rb") as file:
                        shared_weights = file.read()
                
                # Check ratio again
                self.check_step_ratio()

                # Update sample time
                sample_previous = time.time()

            training_end = time.time()

            # Print statistics
            print("Statistics:")
            print(f"Round time: {training_end - check_ratio_start}")
            print(f"Idle time (ratio check): {check_ratio_end - check_ratio_start}")
            print(f"Buffer update time: {buffer_end - check_ratio_end}")
            print(f"Training time: {training_end - buffer_end}")

            # Add statistics to epoch stats (GitHub Copilot wrote this)
            stats_epoch += pd.Series(
                {
                    "memory_len" : len(self.memory),
                    "round_time" : training_end - check_ratio_start,
                    "idle_time" : check_ratio_end - check_ratio_start,
                    **DataFrame(stats_round).mean(skipna=True)
                }
            )

        # Increment epoch
        self.epoch += 1

        # Return epoch statistics
        return stats_epoch

class TrainerMemory:
    # DATA FORMAT:
    # data[0] = indexes
    # data[1] = actions
    # data[2] = observations (images)
    # data[3] = observations (speed)
    # data[4] = observations (gear)
    # data[5] = observations (rpm)
    # data[6] = EOEs (end of episodes - terminated or truncated)
    # data[7] = rewards
    # data[8] = infos
    # data[9] = terminated
    # data[10] = truncated

    def __init__(self, device, steps):
        # Torch Memory Configuration
        self.memory_size = 1000000 # Maximum memory size
        self.device = device # Torch Device

        # Memory Configuration
        self.observation_images = 5 # Number of images to keep track of (same as in GymEnvironment)
        self.action_buffer_length = 2 # Number of actions to keep track of (same as in GymEnvironment)
        self.steps = steps

        # Memory offsets
        self.minimum_samples = max(self.observation_images, self.action_buffer_length)
        self.start_images_offset = max(0, self.minimum_samples - self.observation_images)
        self.start_actions_offset = max(0, self.minimum_samples - self.action_buffer_length)

        # Statistics
        self.test_return_stat = 0.0
        self.train_return_stat = 0.0
        self.test_steps_stat = 0
        self.train_steps_stat = 0

        # Create data
        self.data = []
    
    def __getitem__(self, item):
        # Get transition
        prev_observation, new_action, reward, new_observation, terminated, truncated, info = self.get_transition(item)
        
        # Convert to float32 instead of bool
        terminated = np.float32(terminated)
        truncated = np.float32(truncated)
        
        # Return data
        return prev_observation, new_action, reward, new_observation, terminated, truncated

    def __len__(self):
        # Prevents index out of range errors
        if len(self.data) == 0:
            return 0
        # Only calculate length for first element, as it is a list of all the indices
        length = len(self.data[0]) - self.minimum_samples - 1
        return 0 if length < 0 else length
    
    def __iter__(self):
        for _ in range(self.steps):
            sample = self.sample()
            yield sample
    
    def sample(self):
        indices = (np.random.randint(0, len(self) - 1) for _ in range(256))
        batch = [self[index] for index in indices]
        print("WEEE WOOO")
        batch = collate(batch, self.device)
        return batch
    
    def get_last_eoe(self, eoes):
        for i in reversed(range(len(eoes))):
            if eoes[i]:
                return i
        return None

    def clear_history(history, eoe_index):
        # Make sure index is in range
        if 0 <= eoe_index < len(history):

            # Replace data with first frame of latest episode
            # (easy way is just to go backwards one by one and replace by grabbing the next item)
            # Thanks Leetcode!
            for i in range(len(history)):
                if i <= eoe_index:
                    history[i] = history[i + 1]
    
    def get_transition(self, item):
        # Check if item is an EOE
        if self.data[6][item + self.minimum_samples - 1]:
            # If it is, and it is the first item, increase the item by 1
            if item == 0:
                item += 1
            # If it is, and it is not the first item, decrease the item by 1
            else:
                item -= 1
        
        # Get indices
        index_previous = item + self.minimum_samples - 1
        index_current = index_previous + 1

        # Get actions (into buffers)
        actions = self.load_actions(item)
        last_action_buffer = actions[-1]
        new_action_buffer = actions[1:]

        # Get images (into buffers)
        images = self.load_images(item)
        last_images = images[-1]
        new_images = images[1:]

        # Check if we have multiple EOEs in our buffer
        # If we have multiple EOEs, then our buffer spans multiple episodes
        # So we need to wipe the data before the latest EOE
        last_eoes = self.data[6][index_current - self.minimum_samples : index_current]
        last_eoe_index = self.get_last_eoe(last_eoes)

        # Replace previous data with new data if we find a previous EOE
        if last_eoe_index is not None:
            self.clear_history(new_action_buffer, last_eoe_index - self.start_actions_offset - 1)
            self.clear_history(new_images, last_eoe_index - self.start_images_offset - 1)
            self.clear_history(last_action_buffer, last_eoe_index - self.start_actions_offset)
            self.clear_history(last_images, last_eoe_index - self.start_images_offset)

        # Get data
        new_actions = self.data[1][index_current]
        last_observation = (self.data[2][index_previous], self.data[3][index_previous], self.data[4][index_previous], self.data[5][index_previous], last_images, *last_action_buffer)
        new_observation = (self.data[2][index_current], self.data[3][index_current], self.data[4][index_current], self.data[5][index_current], new_images, *new_action_buffer)
        reward = np.float32(self.data[7][index_current])
        terminated = self.data[9][index_current]
        truncated = self.data[10][index_current]
        info = self.data[8][index_current]

        # Return data
        return last_observation, new_actions, reward, new_observation, terminated, truncated, info
    
    def load_images(self, item):
        # Get images
        images = self.data[2][(item + self.start_images_offset) : (item + self.start_images_offset + self.observation_images + 1)]

        # Convert to numpy stack (and normalize)
        return np.stack(images).astype(np.float32) / 256.0

    def load_actions(self, item):
        # Get actions
        actions = self.data[1][(item + self.start_actions_offset) : (item + self.start_actions_offset + self.action_buffer_length + 1)]

        # Convert to numpy stack (float32)
        return actions

    def append_buffer(self, buffer):
        # Find last element of indices
        first = self.data[0][-1] + 1 if len(self) > 0 else 0

        # Extract data from buffer (indexes)
        indexes = [first + i for i in range(len(buffer))]

        # Extract data from buffer (outputs)
        actions = [element[0] for element in buffer.memory]
        rewards = [element[2] for element in buffer.memory]
        infos = [element[5] for element in buffer.memory]

        # Extract data from buffer (inputs)
        images = [element[1][0] for element in buffer.memory]
        speed = [element[1][1] for element in buffer.memory]
        gear = [element[1][2] for element in buffer.memory]
        rpm = [element[1][3] for element in buffer.memory]

        # Extract data from buffer (eoes)
        terminated = [element[3] for element in buffer.memory]
        truncated = [element[4] for element in buffer.memory]
        eoes = [element[3] or element[4] for element in buffer.memory]

        # Append data (empty list, can't access elements if they don't exist)
        if len(self) == 0: 
            self.data.append(indexes)
            self.data.append(actions)
            self.data.append(images)
            self.data.append(speed)
            self.data.append(gear)
            self.data.append(rpm)
            self.data.append(eoes)
            self.data.append(rewards)
            self.data.append(infos)
            self.data.append(terminated)
            self.data.append(truncated)
        # Append data, non empty list
        else:
            self.data[0] += indexes
            self.data[1] += actions
            self.data[2] += images
            self.data[3] += speed
            self.data[4] += gear
            self.data[5] += rpm
            self.data[6] += eoes
            self.data[7] += rewards
            self.data[8] += infos
            self.data[9] += terminated
            self.data[10] += truncated
        
        # Trim memory if it has exceeded the maximum size
        to_trim = len(self) - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
                
    def append(self, buffer):
        # Make sure buffer is not empty
        if len(buffer) != 0:
            # Update stats
            self.train_return_stat = buffer.train_return_stat
            self.test_return_stat = buffer.test_return_stat
            self.train_steps_stat = buffer.train_steps_stat
            self.test_steps_stat = buffer.test_steps_stat

            # Append buffer
            self.append_buffer(buffer)

