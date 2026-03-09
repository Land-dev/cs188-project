import subprocess
import time
import signal

p = subprocess.Popen(["/Users/yash/miniforge3/envs/drones/bin/python3", "-u", "sim.py"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Let it run for 4 minutes max (180s sim + overhead)
try:
    stdout, _ = p.communicate(timeout=280)
    print(stdout[-3000:])  # print last 3000 chars
except subprocess.TimeoutExpired:
    p.send_signal(signal.SIGINT)
    stdout, _ = p.communicate(timeout=10)
    print("TIMED OUT!")
    print(stdout[-2000:])
