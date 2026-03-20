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
    """Run a single headless sim. Returns (seed, ext, total, elapsed, status, coverage, collisions, loc_error)."""
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
        return (seed, 0, 3, TIMEOUT, "TIMEOUT", 0.0, -1, -1.0)

    elapsed = time.time() - t0

    m = re.search(r"Fires extinguished\s*:\s*(\d+)/(\d+)", output)
    if m:
        ext = int(m.group(1))
        total = int(m.group(2))
        status = "SUCCESS" if ext == total else "PARTIAL"
    else:
        ext, total = 0, 3
        status = "ERROR"
        print(f"\n[Run {seed} ERROR] Last 2000 chars of output:\n{output[-2000:]}")

    mc = re.search(r"Map coverage\s*:\s*([\d.]+)%", output)
    coverage = float(mc.group(1)) if mc else 0.0

    mc_col = re.search(r"Collisions\s*:\s*(\d+)", output)
    collisions = int(mc_col.group(1)) if mc_col else -1

    mc_loc = re.search(r"Avg loc error\s*:\s*([\d.]+)\s*m", output)
    loc_error = float(mc_loc.group(1)) if mc_loc else -1.0

    return (seed, ext, total, elapsed, status, coverage, collisions, loc_error)


if __name__ == "__main__":
    print(f"Running {N} simulations in parallel...")
    print(f"Python: {PYTHON}")
    print(f"Script: {SIM_SCRIPT}")
    print(f"Timeout: {TIMEOUT}s per run")
    print("=" * 65)
    sys.stdout.flush()

    with multiprocessing.Pool(processes=min(N, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_one, range(N))

    print("\n" + "=" * 80)
    print(f"{'Run':<5} {'Status':<10} {'Fires':<10} {'Coverage':>10} {'Collisions':>12} {'Loc Err':>10} {'Time':>8}")
    print("-" * 80)

    successes = 0
    total_coverage = 0.0
    total_collisions = 0
    loc_error_sum = 0.0
    loc_error_count = 0
    for seed, ext, total, elapsed, status, coverage, collisions, loc_error in sorted(results):
        col_str = str(collisions) if collisions >= 0 else "N/A"
        loc_str = f"{loc_error:.4f}m" if loc_error >= 0 else "N/A"
        print(f"{seed:<5} {status:<10} {ext}/{total:<7} {coverage:>9.1f}% {col_str:>12} {loc_str:>10} {elapsed:>7.1f}s")
        if ext == total:
            successes += 1
        total_coverage += coverage
        if collisions >= 0:
            total_collisions += collisions
        if loc_error >= 0:
            loc_error_sum += loc_error
            loc_error_count += 1

    print("-" * 80)
    avg_coverage = total_coverage / N
    avg_time = sum(r[3] for r in results) / N
    avg_loc_error = loc_error_sum / loc_error_count if loc_error_count > 0 else -1.0
    runs_with_collisions = sum(1 for r in results if r[6] > 0)
    print(f"\n  SUCCESS RATE    : {successes}/{N} ({100 * successes / N:.0f}%)")
    print(f"  Avg coverage    : {avg_coverage:.1f}%")
    print(f"  Collision runs  : {runs_with_collisions}/{N} (total contact steps: {total_collisions})")
    if avg_loc_error >= 0:
        print(f"  Avg loc error   : {avg_loc_error:.4f} m")
    print(f"  Avg wall time   : {avg_time:.1f}s")
