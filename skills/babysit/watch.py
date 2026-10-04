#!/usr/bin/env python3
"""Block until a training run needs attention, then print one JSON event and exit.

Stdlib only (Python 3.9+), so it also runs on a cluster login node:
    ssh hpc 'python3 - --log runs/42.out --slurm 42' < watch.py     # one run
    ssh hpc 'python3 - --sweep --glob "runs/*/train_log.jsonl"' < watch.py   # every SLURM job you own

Events: nan, spike, grad_spike, oom, crash, stall, disk, exited (local pid gone), done / died (SLURM),
sweep_done (sweep mode: your queue is empty), and healthy: nothing wrong after --max-minutes, so agent
background-task limits never kill it mid-watch. Restart with --from <offset> (one run) or the same
command (sweep mode keeps its place in --state) to keep watching.
"""
import argparse
import collections
import glob
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
# CANCELLED is deliberately missing: in a sweep, cancelling jobs is usually a person or a script doing it on purpose
SLURM_DIED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


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


class Follower:
    """Follows one log file: new lines go through check(); lines before `quiet_until` only build history."""

    def __init__(self, path, quiet_until, a):
        self.path, self.quiet_until, self.a = path, quiet_until, a
        self.pos, self.buf, self.grew = 0, b"", time.time()
        self.hist = {"loss": collections.deque(maxlen=50), "grad_norm": collections.deque(maxlen=50)}
        self.tail = collections.deque(maxlen=a.tail)

    @property
    def offset(self):
        return self.pos - len(self.buf)  # pass back as --from to resume watching after this point

    def poll(self):
        """(event, line) for the first problem in new output, or None. Stops just past that line."""
        if not os.path.exists(self.path):
            return None
        if os.path.getsize(self.path) < self.pos:  # truncated or rotated
            self.pos, self.buf = 0, b""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            while True:
                loud = self.pos >= self.quiet_until
                chunk = f.read(1 << 20 if loud else min(1 << 20, self.quiet_until - self.pos))
                if not chunk:
                    return None
                self.pos += len(chunk)
                self.grew = time.time()
                *lines, self.buf = (self.buf + chunk).split(b"\n")
                for i, raw in enumerate(lines):
                    for line in raw.decode(errors="replace").split("\r"):
                        if not line.strip():
                            continue
                        self.tail.append(line)
                        event = check(line, self.hist, self.a)
                        if event and loud:
                            # point the offset just past this line so a resume skips it
                            self.pos -= sum(len(r) + 1 for r in lines[i + 1:]) + len(self.buf)
                            self.buf = b""
                            return event, line

    def summary(self):
        losses, grads = self.hist["loss"], self.hist["grad_norm"]
        return {"offset": self.offset,
                "last_loss": losses[-1] if losses else None,
                "last_grad_norm": grads[-1] if grads else None,
                "median_loss": statistics.median(losses) if losses else None,
                "tail": list(self.tail)}


def watch(a):
    started = time.time()
    quiet_until = a.from_offset if a.from_offset is not None else (
        os.path.getsize(a.log) if os.path.exists(a.log) else 0)
    log = Follower(a.log, quiet_until, a)

    def emit(event, detail, state):
        print(json.dumps({"event": event, "detail": detail, "job_state": state} | log.summary(), indent=1))

    while True:
        state = job_state(a)
        found = log.poll()
        if found:
            return emit(found[0], found[1], state)
        if state == "EXITED":
            return emit("exited", "process is gone, check the tail for how it ended", state)
        if state and a.slurm and state not in SLURM_LIVE:
            return emit("done" if state == "COMPLETED" else "died", state, state)
        if state in ("PENDING", "CONFIGURING", "SUSPENDED"):
            log.grew = time.time()
        if time.time() - log.grew > a.stall * 60:
            return emit("stall", "no new log output for %g min" % a.stall, state)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.log))).free / 1e9
        if free < a.min_free:
            return emit("disk", "%.1f GB free where the log is written" % free, state)
        if time.time() - started > a.max_minutes * 60:
            return emit("healthy", "no problems in the last %g min" % a.max_minutes, state)
        if os.getppid() == 1:  # our ssh session or agent died: don't linger on a shared login node
            return None
        time.sleep(a.every)


def sweep(a):
    """Watch every SLURM job you own: each running job's log, any --glob metric files, and how each job ends."""
    started = time.time()
    saved = json.load(open(a.state)) if os.path.exists(a.state) else None
    offsets = saved["offsets"] if saved else {}   # where each log was read up to last time
    known = saved["jobs"] if saved else None      # job id -> name, as of last time
    logs, followers = {}, {}                      # job id -> its SLURM log path; path -> Follower

    def follow(path):
        if path not in followers:
            # first ever run: don't re-alert on what's already in the logs; later runs: anything new is news
            start_at = offsets.get(path, (os.path.getsize(path) if os.path.exists(path) else 0) if saved is None else 0)
            followers[path] = Follower(path, start_at, a)
        return followers[path]

    def emit(event, detail, job=None, log=None, name=None):
        offsets.update({p: f.offset for p, f in followers.items()})
        with open(a.state, "w") as f:
            json.dump({"offsets": offsets, "jobs": known}, f)
        jobs_by_log = {v: k for k, v in logs.items()}
        job = job or jobs_by_log.get(log.path if log else None)
        print(json.dumps({"event": event, "detail": detail, "job": job, "job_name": name or (known or {}).get(job),
                          "log": log.path if log else None, "jobs_left": len(known or {})}
                         | (log.summary() if log else {}), indent=1))

    while True:
        jobs = {}
        for row in sh(["squeue", "-h", "-u", a.user, "-o", "%i|%j|%T"]).splitlines():
            jid, name, state = row.split("|")
            jobs[jid] = (name, state)
        for jid in set(known or {}) - set(jobs):  # left the queue since we last looked
            final = sh(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "State"]).split()
            name = known.pop(jid)
            if final and final[0] in SLURM_DIED:
                known.update({j: n for j, (n, _) in jobs.items()})
                return emit("died", f"job {jid} ({name}) ended {final[0]}", job=jid, name=name)
        known = {j: n for j, (n, _) in jobs.items()}
        if not jobs:
            return emit("sweep_done", "no jobs left in your queue")

        for jid, (name, state) in jobs.items():
            if state == "RUNNING" and jid not in logs:
                out = re.search(r"StdOut=(\S+)", sh(["scontrol", "show", "job", jid]))
                logs[jid] = out.group(1) if out else None
        paths = [p for j, p in logs.items() if p and j in jobs] + sorted(set().union(*[glob.glob(g) for g in a.glob]))
        for path in paths:
            log = follow(path)
            found = log.poll()
            if found:
                return emit(found[0], found[1], log=log)

        if time.time() - started > a.max_minutes * 60:
            return emit("healthy", f"no problems in the last {a.max_minutes:g} min, {len(jobs)} jobs in the queue")
        if os.getppid() == 1:
            return None
        time.sleep(a.every)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", help="training log file to follow (one run)")
    p.add_argument("--sweep", action="store_true", help="watch every SLURM job you own instead of one run")
    p.add_argument("--user", default=os.environ.get("USER", ""), help="SLURM user for --sweep (default: you)")
    p.add_argument("--glob", action="append", default=[],
                   help="--sweep: also follow metric files matching this pattern (repeatable), e.g. 'runs/*/train_log.jsonl'")
    p.add_argument("--state", default=os.path.expanduser("~/.sleeptrain-sweep.json"),
                   help="--sweep: where to remember read positions between restarts")
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
    if not (a.log or a.sweep):
        p.error("give --log (one run) or --sweep (all your SLURM jobs)")
    a.loss_re = re.compile(a.loss_regex, re.I)
    sweep(a) if a.sweep else watch(a)


if __name__ == "__main__":
    main()
