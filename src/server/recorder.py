# Code by Alexandre Haddad-Delaveau
from server import HTTPServer
from enum import Enum
from threading import Thread


class RecordingState(Enum):
    NOT_STARTED = 0
    REQUESTED = 1
    RECORDING = 2
    ENDED = 3

# Recording a replay is complicated...
# - It continously loops, with no clear start / end
#   - So, we find the lowest rpm in the recording and assume that is the start/stop
#   - We should probably find a better solution at some point
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

        while self.recording_state != RecordingState.ENDED:
            pass
    
    def save_replay(self):
        pass

    def server_listener(self):
        # Record lowest RPM if not started
        if self.recording_state == RecordingState.NOT_STARTED:
            if self.lowest_rpm > self.server.rpm:
                self.lowest_rpm = self.server.rpm
        
        # Start recording if requested
        elif self.recording_state == RecordingState.REQUESTED:
            if self.server.rpm <= self.lowest_rpm:
                print(f"Recording started at {self.server.rpm}rpm.")
                self.recording_state = RecordingState.RECORDING

        # If recording, check if the record should stop, otherwise record values
        elif self.recording_state == RecordingState.RECORDING:
            if self.server.rpm <= self.lowest_rpm:
                print(f"Recording ended at {self.server.rpm}rpm. Saving recording...")

                # TODO: Save the recording
                
                # Exit
                print("Recording Saved!")
                self.recording_state = RecordingState.ENDED
            else:
                self.values.append((self.server.pos_x, self.server.pos_y, self.server.pos_z))

                
if __name__ == "__main__":
    ReplayRecorder()