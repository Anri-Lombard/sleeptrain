#!/usr/bin/env python3
"""Block until a training run needs attention, then print one JSON event and exit.

Stdlib only (Python 3.8+), so it also runs on a cluster login node:
    ssh hpc 'python3 - --log runs/42.out --slurm 42' < watch.py

Events: nan, spike, grad_spike, oom, crash, stall, disk, exited (local pid gone), done / died (SLURM),
and healthy: nothing wrong after --max-minutes, so agent background-task limits never kill it mid-watch.
Restart it with --from <offset> to keep watching.
"""
import argparse
import collections
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time

NUM = r"[\s\"':=]*([-+]?(?:\d+\.?\d*(?:e[-+]?\d+)?|nan|inf))"
LOSS = r"(?<![\w/])(?:train[_/])?loss\b" + NUM
GRAD = re.compile(r"grad[_ ]?norm\b" + NUM, re.I)
OOM = re.compile(r"out of memory|OutOfMemoryError|oom[-_ ]?kill|^Killed$", re.I)
CRASH = re.compile(
    r"Traceback \(most recent call last\)|CUDA error|NCCL error|Segmentation fault"
    r"|slurmstepd: error|core dumped",
    re.I,
)
SLURM_LIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def job_state(a):
    """RUNNING/PENDING/... while alive, a final state once gone, None if nothing to track."""
    if a.pid:
        try:
            os.kill(a.pid, 0)
        except ProcessLookupError:
            return "EXITED"
        except PermissionError:
            pass
        return "RUNNING"
    if a.slurm:
        live = sh(["squeue", "-h", "-j", a.slurm, "-o", "%T"])
        if live:
            return live.split()[0]
        final = sh(["sacct", "-n", "-X", "-P", "-j", a.slurm, "-o", "State"])
        return final.split()[0] if final else "GONE"
    return None


def check(line, hist, a):
    """Return an event name if this log line needs attention."""
    if OOM.search(line.strip()):
        return "oom"
    if CRASH.search(line):
        return "crash"
    # grad norm goes NaN or jumps a few steps before the loss does, so it is the early warning
    for name, rx, factor in (("loss", a.loss_re, a.spike), ("grad_norm", GRAD, a.grad_spike)):
        m = rx.search(line)
        if not m:
            continue
        x = float(m.group(1))
        if not math.isfinite(x):
            return "nan"
        # ponytail: ratio-to-median spike test, assumes positive values; RL-style signed losses never trigger it
        h = hist[name]
        med = statistics.median(h) if len(h) >= 10 else 0
        h.append(x)
        if med > 0 and x > factor * med:
            return "spike" if name == "loss" else "grad_spike"
    return None


def watch(a):
    hist = {"loss": collections.deque(maxlen=50), "grad_norm": collections.deque(maxlen=50)}
    losses = hist["loss"]
    started = time.time()
    tail = collections.deque(maxlen=a.tail)
    quiet_until = a.from_offset if a.from_offset is not None else (
        os.path.getsize(a.log) if os.path.exists(a.log) else 0)
    pos, buf, last_growth = 0, b"", time.time()

    def emit(event, detail, state):
        print(json.dumps({
            "event": event,
            "detail": detail,
            "offset": pos - len(buf),  # pass back as --from to resume watching after this point
            "job_state": state,
            "last_loss": losses[-1] if losses else None,
            "last_grad_norm": hist["grad_norm"][-1] if hist["grad_norm"] else None,
            "median_loss": statistics.median(losses) if losses else None,
            "tail": list(tail),
        }, indent=1))

    while True:
        state = job_state(a)
        if os.path.exists(a.log):
            if os.path.getsize(a.log) < pos:  # truncated or rotated
                pos, buf = 0, b""
            with open(a.log, "rb") as f:
                f.seek(pos)
                while True:
                    loud = pos >= quiet_until
                    chunk = f.read(1 << 20 if loud else min(1 << 20, quiet_until - pos))
                    if not chunk:
                        break
                    pos += len(chunk)
                    last_growth = time.time()
                    *lines, buf = (buf + chunk).split(b"\n")
                    for i, raw in enumerate(lines):
                        for line in raw.decode(errors="replace").split("\r"):
                            if not line.strip():
                                continue
                            tail.append(line)
                            event = check(line, hist, a)
                            if event and loud:
                                # point offset just past this line so a resume skips it
                                pos -= sum(len(r) + 1 for r in lines[i + 1:]) + len(buf)
                                buf = b""
                                return emit(event, line, state)

        if state == "EXITED":
            return emit("exited", "process is gone, check the tail for how it ended", state)
        if state and a.slurm and state not in SLURM_LIVE:
            return emit("done" if state == "COMPLETED" else "died", state, state)
        if state in ("PENDING", "CONFIGURING", "SUSPENDED"):
            last_growth = time.time()
        if time.time() - last_growth > a.stall * 60:
            return emit("stall", "no new log output for %g min" % a.stall, state)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.log))).free / 1e9
        if free < a.min_free:
            return emit("disk", "%.1f GB free where the log is written" % free, state)
        if time.time() - started > a.max_minutes * 60:
            return emit("healthy", "no problems in the last %g min" % a.max_minutes, state)
        if os.getppid() == 1:  # our ssh session or agent died: don't linger on a shared login node
            return None
        time.sleep(a.every)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="training log file to follow")
    p.add_argument("--pid", type=int, help="local training process to watch")
    p.add_argument("--slurm", help="SLURM job id to watch")
    p.add_argument("--from", dest="from_offset", type=int,
                   help="byte offset to start alerting from (default: current end of log)")
    p.add_argument("--spike", type=float, default=3.0, help="alert when loss > SPIKE x recent median")
    p.add_argument("--grad-spike", type=float, default=10.0, help="alert when grad norm > this x recent median")
    p.add_argument("--min-free", type=float, default=2.0, help="alert when free disk drops below this many GB")
    p.add_argument("--max-minutes", type=float, default=20,
                   help="exit with a 'healthy' event after this long (stay under your agent's background limit)")
    p.add_argument("--stall", type=float, default=30, help="minutes without log output before alerting")
    p.add_argument("--every", type=float, default=30, help="seconds between polls")
    p.add_argument("--tail", type=int, default=40, help="log lines to include in the event")
    p.add_argument("--loss-regex", default=LOSS, help="regex with one group capturing the train loss")
    a = p.parse_args(argv)
    a.loss_re = re.compile(a.loss_regex, re.I)
    watch(a)


if __name__ == "__main__":
    main()
