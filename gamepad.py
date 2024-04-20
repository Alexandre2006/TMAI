import vgamepad as vg

gamepad = vg.VX360Gamepad()

# Updates gamepad with new values
# Turn (-32768 to 32767)
# Accel (true/false)
# Brake (true/false)
def update(turn, accel, brake):
    # Set new values
    gamepad.left_joystick(turn, 0)
    gamepad.right_trigger(accel*255)
    gamepad.left_trigger(brake*255)

    # Push new values to xinput
    gamepad.update()

# Respawns the car
def respawn():
    gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.update()
    gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.update()

# Resets the gamepad to default values
def reset():
    gamepad.reset()