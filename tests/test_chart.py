"""Run: python3 tests/test_chart.py"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "babysit"))
from chart import first_bad, parse, terminal  # noqa: E402
from watch import LOSS  # noqa: E402

log = "\n".join([
    "{'loss': 2.5, 'step': 1}",                       # HF-style, step after loss on the same line
    "step 2/10 | loss 2.4",
    "  30%|###   | 3/10 [00:30<01:10, 1.0it/s, loss=2.3]",  # tqdm: step 3 of 10, 70 s left
    "val_loss: 9.9",                                  # not the train loss
])
with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
    f.write(log)
pts, grads, total, left = parse(f.name, re.compile(LOSS, re.I))
os.unlink(f.name)
assert [y for _, y in pts] == [2.5, 2.4, 2.3] and pts[-1][0] == 3 and total == 10 and left == 70, (pts, total, left)

steady = [(i, 3 - i * 0.01) for i in range(20)]
assert first_bad(steady) is None
assert first_bad(steady + [(20, 9.0)]) == 20  # same 3x-median rule as the watcher
assert first_bad(steady + [(20, float("nan"))]) == 20
assert "blew up @ step 20" in terminal([("run.log", steady + [(20, 1e6)], [])], 40, None)
assert "⚠ grad norm @ step 20" in terminal([("run.log", steady, [(i, 0.8) for i in range(20)] + [(20, 9.0)])], 40, None)
print("ok")
