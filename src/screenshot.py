# Code by Alexandre Haddad-Delaveau

from PIL import ImageGrab
import time
import os

# X11 Fix (over SSH)
display = None
def get_x_display():
    global display
    if display == None:
        # Command courtesy of [lenroc on Stack Overflow](https://unix.stackexchange.com/questions/17255/is-there-a-command-to-list-all-open-displays-on-a-machine)
        display = os.popen("ps e | grep -Po \"DISPLAY=[\\.0-9A-Za-z:]* \" | sort -u").read().strip().removeprefix("DISPLAY=")
    return display

# Calculates Screenshot FPS
def benchmark(n=1000):
    # Save start time (in ms)
    start = time.perf_counter() * 1000

    # Capture n images
    for _ in range(0,n):
        ImageGrab.grab(xdisplay=get_x_display())
    
    # Collect end time (in ms)
    end = time.perf_counter() * 1000

    # Calculate ms avg. and FPS
    avg = (end-start)/n
    fps = 1/(avg/1000)

    # Print results
    print(f'Screenshot Benchmark: {round(avg, 2)}ms average ({round(fps, 2)} FPS)')

    # Return results
    return avg, fps

def capture():
    return ImageGrab.grab(xdisplay=get_x_display())

if __name__ == "__main__":
    benchmark()

