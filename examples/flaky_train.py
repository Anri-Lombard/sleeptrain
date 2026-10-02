#!/usr/bin/env python3
"""A fake training run that diverges at high learning rates, for trying sleeptrain in 30 seconds.

    python3 examples/flaky_train.py --lr 3e-3 > run.log 2>&1 &   # diverges after step 120, NaN by ~150
    python3 examples/flaky_train.py --lr 1e-3 --resume ckpts/step_100.json > run.r1.log 2>&1 &   # converges
"""
import argparse
import json
import math
import os
import random
import time

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--resume", help="checkpoint to continue from, e.g. ckpts/step_100.json")
p.add_argument("--delay", type=float, default=0.05, help="seconds per step")
a = p.parse_args()

step, loss = 0, 4.0
if a.resume:
    step, loss = json.load(open(a.resume)).values()
    print(f"resumed from {a.resume}", flush=True)
os.makedirs("ckpts", exist_ok=True)

while step < a.steps:
    step += 1
    if a.lr > 2e-3 and step > 120:
        loss = loss * 1.6 if math.isfinite(loss) and loss < 1e6 else float("nan")  # diverges
    else:
        loss = max(1.2, loss * (1 - a.lr * 2)) + random.uniform(-0.03, 0.03)
    print(f"step {step} | loss {loss:.4f} | lr {a.lr:g}", flush=True)
    if step % 50 == 0:
        json.dump({"step": step, "loss": loss}, open(f"ckpts/step_{step}.json", "w"))
        print(f"saved ckpts/step_{step}.json", flush=True)
    time.sleep(a.delay)
print("training complete", flush=True)
