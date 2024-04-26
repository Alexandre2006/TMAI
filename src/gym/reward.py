# Code by Alexandre Haddad-Delaveau
import pickle
import numpy as np

class RewardCalculator:
    def __init__(self):
        # Configuration (Failure Conditions)
        self.max_distance = 100 # Maximum distance from path before failing
        self.max_zero_time = 100 # Maximum time w/o a reward before failing
        self.minimum_failure_time = 100 # Minimum number of frames before failing

        # Configuration (Shortcuts): By checking for future (and previous) points (instead of immediately terminating when we veer off path)
        # we can allow for shortcuts to be made
        self.future_points_considered = 10
        self.previous_points_considered = 10

        # Variables
        self.current_index = 0
        self.current_step = 0
        self.failure_count = 0

        # Load recording
        self.recording = pickle.load(open("recording.pkl", "rb"))
    
    def calculate_reward(self, position):
        # Keep track of whether or not to terminate the episode
        terminate = False

        # Keep track of path index
        index = self.current_index
        best_index = 0
        best_index_distance = np.inf

        # Keep track of shortcuts leniancy
        future_points_considered = self.future_points_considered

        # Increment step counter
        self.current_step += 1 # Just used to check if we should allow failure

        # Calculate distance to nearest point on path
        while index <= len(self.recording):
            distance = np.linalg.norm(position - self.recording[index]) # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            # If this is the best point so far, save it
            if distance < self.best_index_distance:
                best_index = index
                best_index_distance = distance
                future_points_considered = self.future_points_considered
            
            # Increase path index
            index += 1
            future_points_considered -= 1

            # Check if we no longer need to consider future points
            if future_points_considered == 0:
                # Make sure that the direction we are going in is not away from the path (if it is, we should stop moving)
                if best_index_distance > self.max_distance:
                    best_index = self.current_index
                
                # Break out of loop
                break
        
        # Calculate reward (based on how far we got in the track during this step)
        reward = (best_index - self.current_index) / 100.0 # https://www.reddit.com/r/reinforcementlearning/comments/vicory/does_the_value_of_the_reward_matter/
        
        # If we haven't improved, find our previous best index
        # We do this (limited to a certain amount) to allow leniancy for shortcuts
        if best_index == self.current_index:
            # Keep track of best index
            best_index_distance = np.inf
            index = self.current_index

            # Keep track of shortcuts leniancy
            previous_points_considered = self.previous_points_considered

            # Calculate distance to nearest point on path
            while index >= 0:
                distance = np.linalg.norm(position - self.recording[index])
                if distance < best_index_distance:
                    best_index = index
                    best_index_distance = distance
                    previous_points_considered = self.previous_points_considered
                
                # Decrease path index
                index -= 1
                previous_points_considered -= 1

                # Check if we no longer need to consider previous points
                if previous_points_considered == 0:
                    break
            
            # If we've failed too many times, terminate
            if self.current_step > self.minimum_failure_time:
                self.failure_count += 1
                if self.failure_count > self.max_zero_time:
                    terminate = True
        # YAY WE PROGRESSED
        else:
            # Reset failure count
            self.failure_count = 0
        
        # Update current index to be the best index
        self.current_index = best_index

        # Return reward and whether or not to terminate
        return reward, terminate

    def reset(self):
        self.current_index = 0
        self.current_step = 0
        self.failure_count = 0