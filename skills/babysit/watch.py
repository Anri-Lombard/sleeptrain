#!/usr/bin/env python3
"""Block until a training run needs attention, then print one JSON event and exit.

Stdlib only (Python 3.8+), so it also runs on a cluster login node:
    ssh hpc 'python3 - --log runs/42.out --slurm 42' < watch.py

Events: nan, spike, oom, crash, stall, exited (local pid gone), done / died (SLURM).
"""
import argparse
import collections
import json
import math
import os
import re
import statistics
import subprocess
import time

LOSS = r"(?<![\w/])(?:train[_/])?loss\b[\s\"':=]*([-+]?(?:\d+\.?\d*(?:e[-+]?\d+)?|nan|inf))"
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


def check(line, losses, a):
    """Return an event name if this log line needs attention."""
    if OOM.search(line.strip()):
        return "oom"
    if CRASH.search(line):
        return "crash"
    m = a.loss_re.search(line)
    if not m:
        return None
    x = float(m.group(1))
    if not math.isfinite(x):
        return "nan"
    # ponytail: ratio-to-median spike test, assumes positive losses; RL-style signed losses never trigger it
    med = statistics.median(losses) if len(losses) >= 10 else 0
    losses.append(x)
    if med > 0 and x > a.spike * med:
        return "spike"
    return None


def watch(a):
    losses = collections.deque(maxlen=50)
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
                            event = check(line, losses, a)
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
        time.sleep(a.every)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="training log file to follow")
    p.add_argument("--pid", type=int, help="local training process to watch")
    p.add_argument("--slurm", help="SLURM job id to watch")
    p.add_argument("--from", dest="from_offset", type=int,
                   help="byte offset to start alerting from (default: current end of log)")
    p.add_argument("--spike", type=float, default=3.0, help="alert when loss > SPIKE x recent median")
    p.add_argument("--stall", type=float, default=30, help="minutes without log output before alerting")
    p.add_argument("--every", type=float, default=30, help="seconds between polls")
    p.add_argument("--tail", type=int, default=40, help="log lines to include in the event")
    p.add_argument("--loss-regex", default=LOSS, help="regex with one group capturing the train loss")
    a = p.parse_args(argv)
    a.loss_re = re.compile(a.loss_regex, re.I)
    watch(a)


if __name__ == "__main__":
    main()
