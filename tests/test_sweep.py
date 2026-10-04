"""Sweep mode against fake squeue/sacct/scontrol. Run: python3 tests/test_sweep.py"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import time

WATCH = os.path.join(os.path.dirname(__file__), "..", "skills", "babysit", "watch.py")
d = tempfile.mkdtemp()
bin_, state = os.path.join(d, "bin"), os.path.join(d, "state.json")
os.makedirs(bin_)


def fake(name, body):
    path = os.path.join(bin_, name)
    open(path, "w").write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


fake("squeue", f'cat {d}/queue')                     # lines of id|name|state
fake("sacct", f'cat {d}/sacct_$5 2>/dev/null')       # sacct -n -X -P -j <id> ...: $5 is the job id
fake("scontrol", f'echo "StdOut={d}/$3.out"')        # scontrol show job <id>
env = dict(os.environ, PATH=bin_ + os.pathsep + os.environ["PATH"])


def queue(*rows):
    open(f"{d}/queue", "w").write("".join(r + "\n" for r in rows))


def run(after=None, *extra):
    """Start the watcher, optionally change things after it has looked once, return its event."""
    w = subprocess.Popen([sys.executable, WATCH, "--sweep", "--user", "me", "--state", state, "--every", "0.05",
                          "--max-minutes", "0.01", *extra], stdout=subprocess.PIPE, text=True, env=env)
    time.sleep(0.3)
    if after:
        after()
    out, _ = w.communicate(timeout=10)
    return json.loads(out)


open(f"{d}/1.out", "w").write("Traceback (most recent call last):\n")  # an old crash, before we started
open(f"{d}/2.out", "w").write("step 1 | loss 2.0\n")
queue("1|gdn_e4-0|RUNNING", "2|mamba2_e4-0|RUNNING", "3|xlstm_e4-0|PENDING")

# first run: existing log content is history, not news
e = run()
assert e["event"] == "healthy" and e["jobs_left"] == 3, e

# a crash in one job's log is reported with its job id and name
e = run(lambda: open(f"{d}/2.out", "a").write("torch.OutOfMemoryError: CUDA out of memory\n"))
assert (e["event"], e["job"], e["job_name"]) == ("oom", "2", "mamba2_e4-0"), e

# restarting with the same state file doesn't report it again
assert run()["event"] == "healthy"

# a job that left the queue as TIMEOUT wakes us; one that was CANCELLED on purpose does not
open(f"{d}/sacct_1", "w").write("CANCELLED by 1234\n")
open(f"{d}/sacct_2", "w").write("TIMEOUT\n")
queue("3|xlstm_e4-0|RUNNING")
e = run()
assert (e["event"], e["job"], e["job_name"]) == ("died", "2", "mamba2_e4-0") and "TIMEOUT" in e["detail"], e
assert run()["event"] == "healthy"  # job 1's CANCELLED stayed quiet

# metric files from --glob get the loss/grad checks, including files that appear mid-sweep
os.makedirs(f"{d}/runs/a")
open(f"{d}/runs/a/train_log.jsonl", "w").write("".join(json.dumps({"step": i, "loss": 3.0}) + "\n" for i in range(20)))
e = run(lambda: open(f"{d}/runs/a/train_log.jsonl", "a").write(json.dumps({"step": 20, "loss": float("nan")}) + "\n"),
        "--glob", f"{d}/runs/*/train_log.jsonl")
assert e["event"] == "nan" and e["log"].endswith("runs/a/train_log.jsonl"), e

# empty queue: the sweep is over
queue()
open(f"{d}/sacct_3", "w").write("COMPLETED\n")
assert run()["event"] == "sweep_done"
print("ok")
