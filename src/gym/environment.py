# Code by Alexandre Haddad-Delaveau
# w/ based off of Yann Bouteiller's "rtgym"
# https://github.com/yannbouteiller/rtgym

# Helpful docs:
# https://gymnasium.farama.org/api/env

from gymnasium import Env
import gymnasium.spaces as spaces
import time
from collections import deque
from threading import Thread, Lock, Event
from gym.interface import TrackmaniaInterface
import warnings

class GymEnvironment(Env):
    def __init__(self):
        # Interface
        self.__interface = TrackmaniaInterface()
        self.is_waiting = False

        # Configburation (Episodes)
        self.wait_on_completion = True
        self.episode_length = 1000

        # Configuration (Steps)
        self.time_step_interval = 0.05 # (1/20) 20FPS
        self.time_step_timeout = 1.0 # Abort step if taking >1 second
        self.observation_capture_delay = 0.04 # Capture observation near end of time step
        self.default_action = None

        # Observation
        self.__observation_lock = Lock()
        self.__observation = None
        self.__reward = None
        self.__observation_terminated = None
        self.__observation_info = None
        self.__observation_set = False

        # Step Kill Switch
        self.__kill = False

        # Step Information
        self.__step_running = False
        self.__current_action = None
        self.__step_start_time = None
        self.__step_capture_time = None
        self.__step_end_time = None

        # Flags (Threading)
        self.__reset_flag = False
        self.__reset_args = None
        self.__reset_result = None
        self.__wait_flag = False

        # Events (Threading)
        self.__event_logic = Event()
        self.__event_end_reset = Event()
        self.__event_end_wait = Event()
        self.__event_start_step = Event()
        self.__event_end_step = Event()

        # Spaces (Interface)
        self.interface_action_space = None
        self.interface_observation_space = None

        # Background thread
        self._bg_thread = Thread(target=self.__background_worker, args=(), kwargs={}, daemon=True)
        self._bg_thread.start()

        # Start Event Logic
        self.__event_logic.wait()
        self.__event_logic.clear()

        # Spaces
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # Time Management
        self.current_step = 0
        self.time_initialized = False
        self.running = False

        # Gymnasium Variables
        self.seed = None

        # Action Buffer
        self.action_buffer_length = 2
        self.action_buffer = deque(maxlen=self.action_buffer_length)

        # Initialize Action Buffer
        self.initialize_action_buffer()

    def __del__(self):
        self.__kill = True
        self.__event_start_step.set()
        self._bg_thread.join()

    # Time Management
    def _init_time(self):
        # Create a fake timestep
        self.__step_start_time = time.perf_counter()
        self.__step_capture_time = self.__step_start_time + self.observation_capture_delay
        self.__step_end_time = self.__step_start_time + self.time_step_interval

        # Mark time as initialized
        self.time_initialized = True

    def _update_time(self):
        # Get current time
        now = time.perf_counter()

        # If the last timestep has not reached its time limit,
        # we start the next timestep at the same time it ends
        if now < (self.__step_end_time + self.time_step_timeout):
            self.__step_start_time = self.__step_end_time
        else: # Otherwise, set the start time to the current time
            if not self.__wait_flag:
                warnings.warn("Step took too long to complete. Skipping.")
            else:
                self.__wait_flag = False
            self.__step_start_time = now
        
        # Update the expected capture and end times
        self.__step_capture_time = self.__step_start_time + self.observation_capture_delay
        self.__step_end_time = self.__step_start_time + self.time_step_interval

    # Action Buffer
    def initialize_action_buffer(self):
        for _ in range(self.action_buffer_length):
            self.action_buffer.append(self.default_action)

    # Actions + Observations
    def __apply_action(self, action):
        self.__step_running = True
        self.__current_action = action
        self.__event_start_step.set()

    def __apply_observe_wait(self, action):
        # TODO: Preprocess Action Here

        # Send action
        self.__interface.send_control(action)

        # Update timestamps
        self._update_time()

        # Wait until capture time
        now = time.perf_counter()
        if now < self.__step_capture_time:
            time.sleep(self.__step_capture_time - now)
        
        # Capture observation
        self.__observe()

        # Wait until end time
        now = time.perf_counter()
        if now < self.__step_end_time:
            time.sleep(self.__step_end_time - now)
    
    def __observe(self):
        # Acquire observation lock
        self.__observation_lock.acquire()

        # Capture observation
        observation, reward, done, info = self.__interface.get_observation()

        # TODO: Preprocess observation here

        # Set observation
        self.__observation = observation
        self.__reward = reward
        self.__observation_terminated = done
        self.__observation_info = info

        # Set observation flag
        self.__observation_set = True

        # Release observation lock
        self.__observation_lock.release()

    def _wait_for_observation(self):
        waiting = True

        # Wait for observation
        while waiting:
            # Acquire observation lock
            self.__observation_lock.acquire()

            # If an observation is found, we can stop waiting
            if self.__observation_set:
                observation = self.__observation
                reward = self.__reward
                done = self.__observation_terminated
                info = self.__observation_info
                self.__observation_set = False
            
            # Release observation lock
            self.__observation_lock.release()
        
        # Return observation
        return observation, reward, done, info
    
    # Threads
    def _wait_for_worker(self):
        # Verify that a step is running
        if self.__step_running:
            # Wait for the worker to finish
            self.__event_end_step.wait()

            # Clear the event
            self.__event_end_step.clear()

            # Reset the step flag
            self.__step_running = False

    
    def __background_worker(self):
        # Configure default action
        self.default_action = self.__interface.get_default_action()

        # Set spaces
        self.interface_action_space = self.__interface.get_action_space()
        self.interface_observation_space = self.__interface.get_observation_space()

        # Set event logic
        self.__event_logic.set()

        # Main Loop
        while not self.__kill:
            # Wait for next step
            self.__event_start_step.wait()
            self.__event_start_step.clear()

            # Check if reset flag is set
            if self.__reset_flag:
                # Reset seed and options
                seed, options = self.__reset_args
                self.__reset_result = self.__interface.reset(seed=seed, options=options)
                self.__reset_flag = False
                self.__event_end_reset.set()
            # Check for wait flag
            elif self.__wait_flag:
                # Wait for interface to finish
                self.__interface.wait()
                self.__wait_flag = False
                self.__event_end_wait.set()
            # Otherwise, apply action
            else:
                self.__apply_observe_wait(self.__current_action)
            self.__event_end_step.set()

    # Gymnasium Environment Methods
    def step(self, action):
        # Wait for worker to finish
        self._wait_for_worker()

        # Increment step counter
        self.current_step += 1

        # Save action to buffer
        self.action_buffer.append(action)

        # Get result
        if self.running:
            observation, reward, terminated, info = self._wait_for_observation()
        else:
            raise Exception("Environment is not running. Call reset() to start.")

        # Check if finished
        truncated = (self.current_step >= self.episode_length) if not terminated else False
        done = terminated or truncated

        # If not finished, apply action
        if not done:
            self.__apply_action(action)
        # Otherwise, stop the environment
        else:
            self.running = False
            if self.wait_on_completion:
                self.wait()

        # Attach action buffer w/ observation for post-processing
        observation = tuple(*observation, *self.action_buffer)

        # Return observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Wait for worker to finish
        self._wait_for_worker()

        # Configure options
        self.running = True
        self.seed = seed
        self.options = options
        self.current_step = 0

        # Reset action buffer
        self.initialize_action_buffer()

        # Reconfigure args, set flag, and wait for reset
        self.__reset_args = (seed, options)
        self.__reset_flag = True
        self.__step_running = True
        self.__event_start_step.set()
        self.__event_end_reset.wait()
        self.__event_end_reset.clear()

        # Get reset result
        observation, info = self.__reset_result

        # Attach action buffer w/ observation for post-processing
        observation = tuple(*observation, *self.action_buffer)

        # TODO: Run observation pre-processing here

        # Re-initialize time
        if not self.time_initialized:
            self._init_time()
        
        # Wait for worker
        self._wait_for_worker()

        # Run first step
        self.__apply_action(self.action_buffer[-1])
        
        # Return observation & info
        return observation, info

    def stop(self):
        self._wait_for_worker()

    def wait(self):
        self._wait_for_worker()
        self.is_waiting = True
        self.__wait_flag = True
        self.__step_running = True
        self.__event_start_step.set()
        self.__event_end_wait.wait()
        self.__event_end_wait.clear()

    # Spaces
    def _get_action_space(self):
        return self.interface_action_space
    
    def _get_observation_space(self):
        interface_observation_space = self.interface_observation_space
        return spaces.Tuple((*interface_observation_space.spaces, *((self._get_action_space(),) * self.act_buf_len)))