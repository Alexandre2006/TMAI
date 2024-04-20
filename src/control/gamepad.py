# Code by Alexandre Haddad-Delaveau

import vgamepad as vg
import time

class Gamepad():
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
    
    # Resets the gamepad to default values (could be useful if bugs occur)
    def reset(self):
        self.gamepad.reset()
    
    def __press_button(self, button):
        self.gamepad.press_button(button)
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button)
        self.gamepad.update()
    
    # Updates gamepad with new values
    # Turn from -1.0 to 1.0
    # Accel and Brake from 0.0 to 1.0
    def update(self, turn, accel, brake):
        # Set new values
        self.gamepad.right_trigger_float(accel) # Accel
        self.gamepad.left_trigger_float(brake) # Brake
        self.gamepad.left_joystick_float(turn, 0) # Turn

        # Push new values to xinput
        self.gamepad.update()

    # Respawns the car
    def respawn(self):
        # Reset gamepad
        self.reset()

        # Press B
        self.__press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    # Closes the finish screen
    def dismiss_finish(self):
        # Reset gamepad
        self.reset()

        # Press A
        self.__press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    # Saves finish replay for analysis
    def save_finish(self):
        # Reset gamepad
        self.reset()

        # Wait for finish popup to appear
        time.sleep(5)
        
        # Navigate down to save replay
        self.__press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DOWN)

        # Press A to save replay
        self.__press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        # Move back up to close finish screen
        self.__press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_UP)

        # Dismiss finish screen
        self.dismiss_finish()