#!/usr/bin/env python3
"""Show how a run is going: a progress bar and loss sparkline, or an SVG loss chart.

    python3 chart.py run.log run.r1.log                    # two lines for the terminal
    python3 chart.py run.log run.r1.log --svg chart.svg    # chart for the morning report

Pass the logs in order. Each log after the first is a resume and is drawn from the
step it restarted at, so you can see where the run broke and where it picked up.
"""
import argparse
import math
import os
import re
import statistics
from html import escape

from watch import LOSS

STEP = re.compile(r"(?<![\w/])(?:global_)?step\b[\s\"':=]*(\d+)(?:\s*/\s*(\d+))?|(\d+)/(\d+) \[", re.I)
LEFT = re.compile(r"<(?:(\d+):)?(\d+):(\d+)")  # tqdm's time remaining, e.g. [12:01<1:03:22, 3.1it/s]
SPARK = "▁▂▃▄▅▆▇█"


def parse(path, loss_re):
    """[(step, loss)], total steps if the log says, tqdm time left if any."""
    pts, total, left, step = [], None, None, None
    with open(path, errors="replace") as f:
        for raw in f:
            for line in raw.split("\r"):
                s = STEP.search(line)
                if s:
                    step = int(s.group(1) or s.group(3))
                    total = int(s.group(2) or s.group(4) or 0) or total
                t = LEFT.search(line)
                if t:
                    h, m, sec = (int(g or 0) for g in t.groups())
                    left = h * 3600 + m * 60 + sec
                m = loss_re.search(line)
                if m:
                    pts.append((step if step is not None else len(pts), float(m.group(1))))
    return pts, total, left


def first_bad(pts, spike=3.0):
    """Index of the first NaN or blow-up (same rule as watch.py), or None."""
    h = []
    for i, (_, y) in enumerate(pts):
        if not math.isfinite(y) or (len(h) >= 10 and y > spike * statistics.median(h[-50:])):
            return i
        h.append(y)
    return None


def terminal(runs, total, left):
    pts = runs[-1][1]
    if not pts:
        return "no loss lines yet in " + runs[-1][0]
    step, loss = pts[-1]
    if total:
        frac = min(1.0, step / total)
        bar = "█" * round(24 * frac) + "░" * (24 - round(24 * frac))
        head = f"{runs[-1][0]}  [{bar}]  {frac:4.0%}  step {step:,}/{total:,}"
    else:
        head = f"{runs[-1][0]}  step {step:,}"
    if left is not None:
        head += f"  · ~{left // 3600}h {left % 3600 // 60:02d}m left"
    c = first_bad(pts)
    finite = [y for _, y in pts[:c]][-240:]  # sparkline of the healthy part, a 1e6 blow-up would flatten it
    spark = ""
    if finite:
        k = max(1, len(finite) // 30)
        buckets = [statistics.mean(finite[i:i + k]) for i in range(0, len(finite), k)]
        lo, hi = min(buckets), max(buckets)
        spark = "".join(SPARK[int((b - lo) / (hi - lo or 1) * 7.999)] for b in buckets)
    tail = f"  min {min(finite):.4g}" if finite else ""
    if c is not None:
        tail += f" · ✕ {'NaN' if not math.isfinite(pts[c][1]) else 'blew up'} @ step {pts[c][0]:,}"
    if len(runs) > 1:
        tail += f" · {len(runs) - 1} resume{'s' * (len(runs) > 2)}"
    return f"{head}\nloss {loss:.4g}  {spark}{tail}"


def svg(runs, total):
    W, H, L, R, T, B = 760, 340, 64, 24, 64, 46
    cut = [first_bad(pts) for _, pts in runs]
    healthy = [y for (_, pts), c in zip(runs, cut) for _, y in pts[:c]]
    if not healthy:
        return None
    xmax = max([total or 0] + [x for _, pts in runs for x, _ in pts]) or 1
    lo, hi = min(healthy), max(healthy)
    hi = hi if hi > lo else lo + 1
    X = lambda x: L + (W - L - R) * x / xmax
    Y = lambda y: T + (H - T - B) * (1 - (min(y, hi) - lo) / (hi - lo)) if math.isfinite(y) else T - 4
    font = 'font-family="ui-monospace, Menlo, monospace"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" rx="12" fill="#191817"/>',
         f'<text x="{L}" y="34" {font} font-size="15" fill="#8b8783">loss · {escape(runs[0][0])}</text>']
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#2e2c2a"/>')
        o.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" {font} font-size="11" fill="#8b8783">{v:.3g}</text>')
    o.append(f'<text x="{L}" y="{H - 18}" {font} font-size="11" fill="#8b8783">0</text>')
    o.append(f'<text x="{W - R}" y="{H - 18}" text-anchor="end" {font} font-size="11" fill="#8b8783">step {xmax:,}</text>')
    for n, ((_, pts), c) in enumerate(zip(runs, cut)):
        if not pts:
            continue
        last = n == len(runs) - 1
        if n:
            x0 = X(pts[0][0])
            o.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{T - 6}" y2="{H - B}" stroke="#d97757" stroke-dasharray="3 4" opacity=".7"/>')
            o.append(f'<text x="{x0 + 6:.1f}" y="{H - B - 8}" {font} font-size="11" fill="#d97757">resumed @ {pts[0][0]:,}</text>')
        shown = pts if c is None else pts[:c + 1]  # past a blow-up the curve is just the clip line
        k = max(1, len(shown) // 600)
        line = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in shown[::k] + shown[-1:])
        style = 'stroke="#d97757" stroke-width="2.5"' if last else 'stroke="#8b8783" stroke-width="2" stroke-dasharray="5 4"'
        o.append(f'<polyline points="{line}" fill="none" {style} stroke-linejoin="round"/>')
        if c is not None:
            bx, by = pts[c]
            label = "NaN" if not math.isfinite(by) else "blew up"
            o.append(f'<text x="{X(bx):.1f}" y="{T - 10}" text-anchor="middle" {font} font-size="12" font-weight="bold" fill="#ff6b6b">✕ {label} @ {bx:,}</text>')
    step, loss = runs[-1][1][-1] if runs[-1][1] else (0, float("nan"))
    broke = cut[-1] is not None
    o.append(f'<text x="{W - R}" y="36" text-anchor="end" {font} font-size="22" font-weight="bold" '
             f'fill="{"#ff6b6b" if broke else "#e8e6e3"}">{"NaN" if not math.isfinite(loss) else f"{loss:.4g}"}</text>')
    if total:
        frac = min(1.0, step / total)
        o.append(f'<rect x="{L}" y="{H - 12}" width="{W - L - R}" height="4" rx="2" fill="#2e2c2a"/>')
        o.append(f'<rect x="{L}" y="{H - 12}" width="{(W - L - R) * frac:.1f}" height="4" rx="2" fill="{"#ff6b6b" if broke else "#7ee787"}"/>')
    o.append("</svg>")
    return "\n".join(o)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="the run's logs, first run first, then each resume")
    p.add_argument("--svg", help="write an SVG loss chart here instead of printing")
    p.add_argument("--total", type=int, help="total steps, if the log doesn't print step N/TOTAL")
    p.add_argument("--loss-regex", default=LOSS)
    a = p.parse_args(argv)
    loss_re = re.compile(a.loss_regex, re.I)
    runs, total, left = [], a.total, None
    for path in a.logs:
        pts, t, left = parse(path, loss_re)
        runs.append((os.path.basename(path), pts))
        total = a.total or t or total
    if a.svg:
        out = svg(runs, total)
        if out is None:
            raise SystemExit("no finite loss values to plot yet")
        open(a.svg, "w").write(out)
        print("wrote", a.svg)
    else:
        print(terminal(runs, total, left))


if __name__ == "__main__":
    main()
