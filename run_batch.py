"""Run N headless simulations in parallel and report success rate + coverage."""
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
    """Run a single headless simulation.

    Returns (seed, fires_extinguished, total_fires, elapsed, status, coverage_pct, wall_time).
    """
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
        return (seed, 0, 3, TIMEOUT, "TIMEOUT", 0.0)

    elapsed = time.time() - t0

    # Parse "  Fires extinguished : X/Y"  (current format)
    m = re.search(r"Fires extinguished\s*:\s*(\d+)/(\d+)", output)
    if m:
        ext = int(m.group(1))
        total = int(m.group(2))
        status = "SUCCESS" if ext == total else "PARTIAL"
    else:
        ext, total = 0, 3
        status = "ERROR"
        # Show the tail of the output where errors typically appear
        print(f"\n[Run {seed} ERROR] Last 2000 chars of output:\n{output[-2000:]}")

    # Parse "  Map coverage       : XX.X%"
    mc = re.search(r"Map coverage\s*:\s*([\d.]+)%", output)
    coverage = float(mc.group(1)) if mc else 0.0

    return (seed, ext, total, elapsed, status, coverage)


if __name__ == "__main__":
    print(f"Running {N} simulations in parallel...")
    print(f"Python: {PYTHON}")
    print(f"Script: {SIM_SCRIPT}")
    print(f"Timeout: {TIMEOUT}s per run")
    print("=" * 65)
    sys.stdout.flush()

    with multiprocessing.Pool(processes=min(N, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_one, range(N))

    print("\n" + "=" * 65)
    print(f"{'Run':<5} {'Status':<10} {'Fires':<10} {'Coverage':>10} {'Time':>8}")
    print("-" * 45)

    successes = 0
    total_coverage = 0.0
    for seed, ext, total, elapsed, status, coverage in sorted(results):
        print(f"{seed:<5} {status:<10} {ext}/{total:<7} {coverage:>9.1f}% {elapsed:>7.1f}s")
        if ext == total:
            successes += 1
        total_coverage += coverage

    print("-" * 45)
    avg_coverage = total_coverage / N
    avg_time = sum(r[3] for r in results) / N
    print(f"\n  SUCCESS RATE : {successes}/{N} ({100 * successes / N:.0f}%)")
    print(f"  Avg coverage : {avg_coverage:.1f}%")
    print(f"  Avg wall time: {avg_time:.1f}s")
