# Code by Alexandre Haddad-Delaveau

from pathlib import Path
import control.screenshot as screenshot
from trainer.trainer import Trainer
from actor.actor import Actor
import threading

# Create two threads, one with an actor and another with a trainer, running in parallel

def start_actor():
    actor = Actor()
    actor.run(1)

def start_trainer():
    trainer = Trainer()
    trainer.run_epoch()

def main():
    # Create two seperate threads (running in parallel) for trainer and actor
    #trainer_thread = threading.Thread(target=start_trainer)
    #trainer_thread.start()
    start_actor()
    start_trainer()

if __name__ == "__main__":
    main()    