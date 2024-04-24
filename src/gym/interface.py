# Code by Alexandre Haddad-Delaveau
# Interface between Trackmania (vision input / gamepad output) & Gym environment

import gymnasium.spaces as spaces
from control.gamepad import Gamepad
from collections import deque
import time
import screenshot
import cv2
import numpy as np

class TrackmaniaInterface():
    def __init__(self):
        # Image Options
        self.img_size = (64, 64)
        self.img = None
        self.img_hist = deque(5) # Try 5 for now, adjust later 
        
        # Reward / Penalty Parameters
        self.finish_reward = 100
        self.constant_penalty = 0
        self.reward_calculator = None # TODO: Make reward calculating class
        
        # Time tracking
        self.last_time = None
        
        # Gamepad Inputs
        self.gamepad = Gamepad()        
    
    def send_control(self, control):
        self.gamepad.update(control[0], control[1], control[2])
        
    def reset(self):
        # Reset controller and restart the race
        self.gamepad.respawn()
        
        # Wait for game to refresh
        time.sleep(1.5)
        
        # Capture initial observation
        img = self.observe()
        
        # Reset image history
        for _ in range(len(self.img_hist)):
            self.img_hist.append(img)
        
        # TODO: Reset Reward Function
        
        # Return initial observation
        return img
    
    def observe(self):
        img = screenshot.capture()
        
        # Resize image to smaller size for improved performance
        img = cv2.resize(img, self.img_size)
        
        # Remove color from image (does not provide any info)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save image
        self.img = img
        
        # Return observation
        return img
        
    def wait(self):
        # Just reset race, no real way to wait
        self.gamepad.respawn()
    
    def get_observation_data(self):
        # Get observation
        img = self.observe()
        
        # Add image to image history
        self.img_hist.append(img)
        
        # Calculate reward TODO: calculate reward
        reward, terminated = None
        
        # Update reward with penalty
        reward += self.constant_penalty
        
        # TODO: Way to check if track has ended
        
        # Convert reward to float32
        reward = np.float32(reward)
        
        # Return observation (info can be left blank)
        return img, reward, terminated, {}

    def get_observation_space(self):
        # Return observation (0-255 due to grayscale)
        return spaces.Box(low=0.0, high=255.0, shape=(len(self.img_hist), self.img_size[1], self.img_size[0]))
    
    def get_action_space(self):
        # Range from -1.0 to 1.0 as this is the range for steering.
        # Braking and acceleration are off when <= 0 and on when > 0
        return spaces.Box(low=1.0, high=1.0, shape=(3, ))
    
    def get_default_action(self):
        # Don't do anything for default action
        return np.array([0.0, 0.0, 0.0], dtype='float32')