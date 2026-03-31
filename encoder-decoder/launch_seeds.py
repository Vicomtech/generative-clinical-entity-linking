#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launches medical_t5_mapped_context_seeds.py in parallel,
one process per seed, each pinned to a dedicated GPU.
"""

import subprocess
import sys
import os

SCRIPT = os.path.join(os.path.dirname(__file__), 'medical_t5_mapped_context_seeds.py')
PYTHON  = sys.executable   # same interpreter / conda env that runs this launcher

# seed → GPU index (adjust GPU assignments if needed)
RUNS = [
    # {"seed": 42,  "gpu": 0},
    # {"seed": 123, "gpu": 1},
    {"seed": 456, "gpu": 3},
]

processes = {}
for run in RUNS:
    seed, gpu = run["seed"], run["gpu"]
    cmd = [PYTHON, SCRIPT, "--seed", str(seed)]

    # Build a clean env with CUDA_VISIBLE_DEVICES set before the process starts
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = f"logs/run_seed{seed}_gpu{gpu}.log"
    os.makedirs("logs", exist_ok=True)
    log_file = open(log_path, "w")

    print(f"Launching seed={seed} on GPU {gpu}  →  log: {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    processes[seed] = (proc, log_file, log_path)

print("\nAll 3 runs started. Waiting for completion ...\n")

exit_codes = {}
for seed, (proc, log_file, log_path) in processes.items():
    proc.wait()
    log_file.close()
    exit_codes[seed] = proc.returncode
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"  seed={seed}  {status}  —  log: {log_path}")

print("\nDone.")
if any(code != 0 for code in exit_codes.values()):
    sys.exit(1)
