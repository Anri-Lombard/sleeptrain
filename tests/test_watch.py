"""Run: python3 tests/test_watch.py"""
import json
import os
import subprocess
import sys
import tempfile
import time

WATCH = os.path.join(os.path.dirname(__file__), "..", "skills", "babysit", "watch.py")


def run(history, new, *args):
    """Log `history`, start the watcher, then append `new`; return the event it prints."""
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "run.log")
        open(log, "w").write("".join(l + "\n" for l in history))
        w = subprocess.Popen([sys.executable, WATCH, "--log", log, "--every", "0.05", *args],
                             stdout=subprocess.PIPE, text=True)
        time.sleep(0.3)
        with open(log, "a") as f:
            f.write("".join(l + "\n" for l in new))
        out, _ = w.communicate(timeout=10)
        return json.loads(out)


losses = [f"step {i} | loss {3 - i * 0.01:.3f}" for i in range(20)]

# old problems before the watcher started don't fire; new ones do, with history kept for the spike test
e = run(["loss: nan", "CUDA out of memory"] + losses, ["step 20 | loss 9.5"])
assert e["event"] == "spike" and e["last_loss"] == 9.5, e

assert run(losses, ["{'loss': nan, 'grad_norm': inf}"])["event"] == "nan"  # HF Trainer dict
assert run(losses, ["100%|###| loss=NaN"])["event"] == "nan"  # tqdm postfix
assert run(losses, ["{'loss': 2.9, 'grad_norm': nan}"])["event"] == "nan"  # grads go first
grads = [f"step {i} | loss 2.0 | grad_norm 0.8" for i in range(20)]
assert run(grads, ["step 20 | loss 2.0 | grad_norm 9.1"])["event"] == "grad_spike"
assert run([], ["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"])["event"] == "oom"
assert run([], ["Traceback (most recent call last):", '  File "train.py"'])["event"] == "crash"

# a slow climb never trips the spike test but does trip drift, and only once: a restart replays the log
falls = [f"step {i} | loss {3 - i * 0.03:.3f}" for i in range(70)]
creep = [f"step {70 + i} | loss {0.9 * 1.01 ** i:.3f}" for i in range(120)]
e = run(falls, creep)
assert e["event"] == "drift", e
with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "run.log")
    open(log, "w").write("".join(l + "\n" for l in falls + creep + ["step 999 | loss 1.9"]))
    again = subprocess.run([sys.executable, WATCH, "--log", log, "--from", str(e["offset"]), "--every", "0.05",
                            "--max-minutes", "0.01"], capture_output=True, text=True, timeout=10)
    assert json.loads(again.stdout)["event"] == "healthy", again.stdout

# look-alikes that must not fire: val loss, loss scale, a single noisy step
quiet = losses + ["val_loss: nan", "reducing loss scale to 32768", "step 21 | loss 3.4", "loss_scale: 65536"]
e = run([], quiet + ["Killed"])
assert e["event"] == "oom" and e["detail"] == "Killed", e

# resuming from the reported offset skips the event that was already handled
with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "run.log")
    open(log, "w").write("loss: nan\nstep 1 | loss 2.0\nTraceback (most recent call last):\n")
    first = subprocess.run([sys.executable, WATCH, "--log", log, "--from", "0", "--every", "0.05"],
                           capture_output=True, text=True, timeout=10)
    e = json.loads(first.stdout)
    assert e["event"] == "nan", e
    again = subprocess.run([sys.executable, WATCH, "--log", log, "--from", str(e["offset"]), "--every", "0.05"],
                           capture_output=True, text=True, timeout=10)
    assert json.loads(again.stdout)["event"] == "crash"

# a watched local process that exits is reported
p = subprocess.Popen([sys.executable, "-c", "pass"])
p.wait()
with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "run.log")
    out = subprocess.run([sys.executable, WATCH, "--log", log, "--pid", str(p.pid), "--every", "0.05"],
                         capture_output=True, text=True, timeout=10).stdout
    assert json.loads(out)["event"] == "exited"

# with nothing wrong it checks in as healthy instead of running forever
with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "run.log")
    open(log, "w").write("step 1 | loss 2.0\n")
    out = subprocess.run([sys.executable, WATCH, "--log", log, "--every", "0.05", "--max-minutes", "0.005"],
                         capture_output=True, text=True, timeout=10).stdout
    assert json.loads(out)["event"] == "healthy"

print("ok")
