#!/usr/bin/env python3
"""A small fake training run that saves a checkpoint every 50 steps, for trying sleeptrain.

    python3 examples/flaky_train.py --lr 3e-3 > run.log 2>&1 &
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

# Gradient descent on a quadratic bowl whose curvature grows as training goes on, as the sharpness of
# real networks tends to. Each step multiplies the parameter by (1 - lr * curvature).
random.seed(0)
step, theta = 0, 1e-3
if a.resume:
    ck = json.load(open(a.resume))
    step, theta = ck["step"], ck["theta"]
    print(f"resumed from {a.resume}", flush=True)
os.makedirs("ckpts", exist_ok=True)

while step < a.steps:
    step += 1
    curvature = 430 + 2 * step
    theta = (1 - a.lr * curvature) * theta + random.gauss(0, 1e-3)
    if not math.isfinite(theta) or abs(theta) > 1e150:
        theta = float("nan")
    loss = 1.2 + 2.8 * math.exp(-step / 120) + curvature * theta ** 2 / 2 + random.uniform(-0.03, 0.03)
    grad_norm = curvature * abs(theta) + random.uniform(0, 0.2)
    print(f"step {step}/{a.steps} | loss {loss:.4f} | grad_norm {grad_norm:.2f} | lr {a.lr:g}", flush=True)
    if step % 50 == 0:
        json.dump({"step": step, "theta": theta, "loss": loss}, open(f"ckpts/step_{step}.json", "w"))
        print(f"saved ckpts/step_{step}.json", flush=True)
    time.sleep(a.delay)
print("training complete", flush=True)
