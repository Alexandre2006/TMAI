# Code by Alexandre Haddad-Delaveau

import control.screenshot as screenshot
import logging

for i in range(0,10):
    print(f"RUN {i+1}: ", end="")
    screenshot.benchmark() 


# Hey, this file is going to be blank for quite a while longer...
# So here are some notes for what I need to do:
# 1. Implement SAC (DONE)
# 2. Create the actual actor (who will play the game and send the data back to the trainer)
# 3. Build connection between actor and trainer (DONE)
# 4. Build the main script

# So... I've forgotten this twice today: Why are we seperating the training and the acting?
# Is it a) because we want to have multiple actors?
# b) because we want multiple trainers?
# or c) because we want to be able to train on a different machine than we play on?

# If you said C, you where the closest! Congrats!!!!
# We don't exactly want to play on a different machine, the bigger issue is that we can't exactly train at real time speeds (I think).
# We need to instead build a model, send it off to actor, wait for actor to play, and then send it back to the trainer.