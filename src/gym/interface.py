# Code by Alexandre Haddad-Delaveau
# Interface between Trackmania (vision input / gamepad output) & Gym environment

import gymnasium.spaces as spaces
from control.gamepad import Gamepad
from collections import deque
import time
import control.screenshot as screenshot
import cv2
import numpy as np
from gym.reward import RewardCalculator
from server.server import HTTPServer

class TrackmaniaInterface():
    def __init__(self):
        # Image Options
        self.img_size = (64, 64)
        self.img = None
        self.img_hist = deque(maxlen=5) # Try 5 for now, adjust later 
        
        # Reward / Penalty Parameters
        self.finish_reward = 100
        self.constant_penalty = 0
        self.reward_calculator = RewardCalculator()
        
        # Time tracking
        self.last_time = None
        
        # Gamepad Inputs
        self.gamepad = Gamepad()        

        # Server
        self.server = HTTPServer()
    
    def send_control(self, control):
        self.gamepad.update(control[0], control[1], control[2])
        
    def reset(self):
        # Reset controller and restart the race
        self.gamepad.respawn()
        
        # Wait for game to refresh
        time.sleep(1.5)
        
        # Capture initial observation
        img, pos = self.observe()
        
        # Reset image history
        for _ in range(len(self.img_hist)):
            self.img_hist.append(img)
        
        # Reset Reward Function
        self.reward_calculator.reset()

        # Convert types to numpy
        speed = np.array([
            self.server.speed,
        ], dtype='float32')
        gear = np.array([
            self.server.gear,
        ], dtype='float32')
        rpm = np.array([
            self.server.rpm,
        ], dtype='float32')
        for i in range(len(self.img_hist)):
            self.img_hist.append(img)
        images = np.array(list(self.img_hist))

        print("GOT OBSERVATION FROMR ESET")
        print(images)

        # Return initial observation # TODO: FIX
        return [np.array([]), speed, gear, rpm], {}
    
    def observe(self):
        img = screenshot.capture()

        # Convert to numpy array
        img = np.array(img)
        
        # Resize image to smaller size for improved performance
        img = cv2.resize(img, self.img_size)
        
        # Remove color from image (does not provide any info)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save image
        self.img = img

        # Get positional data
        pos = (self.server.pos_x, self.server.pos_y, self.server.pos_z)
        
        # Return observation
        return img, pos
        
    def wait(self):
        # Just reset race, no real way to wait
        self.gamepad.respawn()
    
    def get_observation_data(self):
        # Get observation
        img, pos = self.observe()
        
        # Add image to image history
        self.img_hist.append(img)
        
        # Calculate reward
        reward, terminated = self.reward_calculator.calculate_reward(pos)
        
        # Update reward with penalty
        reward += self.constant_penalty
        
        # Check if track has ended
        ended = self.server.racing == False
        if ended:
            reward += self.finish_reward
            terminated = True
        
        # Convert reward to float32
        reward = np.float32(reward)

        print("GOT OBSERVATION FROM MAIN COMMAND")
        
        # Return observation (info can be left blank)
        return np.array(list(self.img_hist)), reward, terminated, {}

    def get_observation_space(self):
        # Return observation (0-255 due to grayscale)
        img = spaces.Box(low=0.0, high=255.0, shape=(len(self.img_hist), self.img_size[1], self.img_size[0]))
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6.0, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=11000.0, shape=(1, ))
        return spaces.Tuple((img, speed, gear, rpm))
    
    def get_action_space(self):
        # Range from -1.0 to 1.0 as this is the range for steering.
        # Braking and acceleration are off when <= 0 and on when > 0
        return spaces.Box(low=1.0, high=1.0, shape=(3, ))
    
    def get_default_action(self):
        # Don't do anything for default action
        return np.array([0.0, 0.0, 0.0], dtype='float32')