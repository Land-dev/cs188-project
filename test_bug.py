import subprocess
import time

def run():
    p = subprocess.Popen(["conda", "run", "-n", "drones", "python3", "sim.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = ""
    for line in iter(p.stdout.readline, ''):
        print(line, end='')
        out += line
        if "Extinguished" in line or "extinguished" in line:
            pass
        if "SUCCESS" in line or "RESULTS:" in line:
            break
        if "BaseAviary" in line:
            pass
    p.stdout.close()
    p.wait()

run()
