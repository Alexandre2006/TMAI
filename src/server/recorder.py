# Code by Alexandre Haddad-Delaveau
from server import HTTPServer
from enum import Enum
from threading import Thread
import numpy as np
import pickle
import time

def save_recording(recording):
    positions = np.array(recording)
    # Increase smoothness by generating more points
    # Two strategies:
    #  1. Double points, loop n times
    #  2. Generate points in between until maximum distance is reached (seems like it would make a smoother line if there are big differences in distance between points)
    max_distance = 0.1
    final_positions = np.array(positions[0]) # Copy first position
    i = 0

    # Get the last point
    last_point = final_positions[-1]

    while i < len(positions):
        # Get the next point
        next_point = positions[i]

        # Generate new point
        distance = np.linalg.norm(next_point - last_point) # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        if distance < max_distance:
            # Move onto next point (since we can not generate any more points here)
            last_point = next_point
            i += 1
            print(f"{round(i/len(positions), 3) * 100}% ({i}/{len(positions)})")

            # Append point
            final_positions = np.append(final_positions, next_point)
        else:
            # Calculate unit vector
            unit_vector = (next_point - last_point) / distance

            # Calculate coordinates of point
            new_point = last_point + (unit_vector * max_distance)

            # Append point
            final_positions = np.append(final_positions, new_point)

            # Update last point
            last_point = new_point
        
    # Save the recording using pickle because I'm lazy
    pickle.dump(final_positions, open("recording.pkl", "wb"))

class RecordingState(Enum):
    NOT_STARTED = 0
    REQUESTED = 1
    RECORDING = 2
    ENDED = 3
    SAVED = 4

class ReplayRecorder:
    def __init__(self):
        # Init Variables
        self.lowest_rpm = 11000 # Seems to be the highest RPM in the game (that I've been able to reach)
        self.recording_state = RecordingState.NOT_STARTED
        self.values = []

        # Start server listener
        self.server = HTTPServer(listeners=[self.server_listener])

        # Print Instructions
        print("Speed monitoring started. Please wait for the replay to complete a full cycle.")
        input("Ctrl+C to exit. Hit enter to start recording...")

        # Start recording
        print(f"Recording start requested. Waiting for lowest RPM ({self.lowest_rpm}rpm) to start recording...")
        self.recording_state = RecordingState.REQUESTED

        while self.recording_state != RecordingState.SAVED:
            time.sleep(1)
        
    def server_listener(self):
        # Record lowest RPM if not started
        if self.recording_state == RecordingState.NOT_STARTED:
            if self.lowest_rpm > self.server.rpm and self.server.rpm != 0:
                self.lowest_rpm = self.server.rpm
        
        # Start recording if requested
        elif self.recording_state == RecordingState.REQUESTED:
            if self.server.rpm <= self.lowest_rpm:
                print(f"Recording started at {self.server.rpm}rpm.")
                self.recording_state = RecordingState.RECORDING

        # If recording, check if the record should stop, otherwise record values
        elif self.recording_state == RecordingState.RECORDING:
            if self.server.rpm <= self.lowest_rpm:
                self.recording_state = RecordingState.ENDED
                print(f"Recording ended at {self.server.rpm}rpm. Saving recording...")

                # Save the recording
                save_recording(self.values)
                
                # Exit
                print("Recording Saved!")
                self.recording_state = RecordingState.SAVED
            else:
                self.values.append((self.server.pos_x, self.server.pos_y, self.server.pos_z))

class LiveRecorder:
    def __init__(self):
        # Init Variables
        self.recording_state = RecordingState.NOT_STARTED
        self.values = []

        # Start server listener
        self.server = HTTPServer(listeners=[self.server_listener])

        # Print Instructions
        input("Ctrl+C to exit. Hit enter to start recording...")

        # Start recording
        print(f"Recording start requested. Waiting for player to start racing...")
        self.recording_state = RecordingState.REQUESTED

        while self.recording_state != RecordingState.SAVED:
            time.sleep(1)

    def server_listener(self):
        # Ignore if not started
        if self.recording_state == RecordingState.NOT_STARTED:
            pass
        
        # Start recording if requested
        elif self.recording_state == RecordingState.REQUESTED:
            if self.server.racing and self.server.rpm != 0:
                print(f"Recording started.")
                self.recording_state = RecordingState.RECORDING

        # If recording, check if the record should stop, otherwise record values
        elif self.recording_state == RecordingState.RECORDING:
            if not self.server.racing or self.server.rpm == 0:
                self.recording_state = RecordingState.ENDED
                print(f"Recording ended. Saving recording...")

                # Save the recording
                save_recording(self.values)
                
                # Exit
                print("Recording Saved!")
                self.recording_state = RecordingState.SAVED
            else:
                self.values.append((self.server.pos_x, self.server.pos_y, self.server.pos_z))

                
if __name__ == "__main__":
    response = input("Would you like to record a replay (r) or a live recording (l)? ")
    if response == "r":
        ReplayRecorder()
    elif response == "l":
        LiveRecorder()
    else:
        print("Invalid response. Exiting...")