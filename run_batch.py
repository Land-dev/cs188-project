"""Run N headless simulations in parallel and report success rate."""
import subprocess
import multiprocessing
import time
import re
import sys
import os

N = 10
PYTHON = sys.executable  # same interpreter as the one running this script
SIM_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim.py")
TIMEOUT = 300  # 5 min max per run


def run_one(seed):
    """Run a single headless simulation. Returns (seed, fires_extinguished, total_fires, elapsed)."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SIM_SEED"] = str(seed)
    
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "-u", SIM_SCRIPT, "--headless"],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return (seed, 0, 3, TIMEOUT, "TIMEOUT")
    
    elapsed = time.time() - t0
    
    # Parse "RESULTS: X/Y fires extinguished"
    m = re.search(r"RESULTS:\s+(\d+)/(\d+)\s+fires extinguished", output)
    if m:
        ext = int(m.group(1))
        total = int(m.group(2))
        status = "SUCCESS" if ext == total else "PARTIAL"
    else:
        ext, total = 0, 3
        status = "ERROR"
        # Print the error so we can debug headless failures!
        print(f"\n[Run {seed} Failed] Error output:\n{output[:1000]}")
    
    return (seed, ext, total, elapsed, status)


if __name__ == "__main__":
    print(f"Running {N} simulations in parallel...")
    print(f"Python: {PYTHON}")
    print(f"Script: {SIM_SCRIPT}")
    print(f"Timeout: {TIMEOUT}s per run")
    print("=" * 60)
    sys.stdout.flush()
    
    with multiprocessing.Pool(processes=min(N, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_one, range(N))
    
    print("\n" + "=" * 60)
    print(f"{'Run':<5} {'Status':<10} {'Fires':<10} {'Time':>8}")
    print("-" * 35)
    
    successes = 0
    for seed, ext, total, elapsed, status in sorted(results):
        print(f"{seed:<5} {status:<10} {ext}/{total:<7} {elapsed:>7.1f}s")
        if ext == total:
            successes += 1
    
    print("-" * 35)
    print(f"\nSUCCESS RATE: {successes}/{N} ({100*successes/N:.0f}%)")
    print(f"Average time: {sum(r[3] for r in results)/N:.1f}s")
