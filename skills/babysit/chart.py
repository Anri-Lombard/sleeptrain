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

from watch import GRAD, LOSS

STEP = re.compile(r"(?<![\w/])(?:global_)?step\b[\s\"':=]*(\d+)(?:\s*/\s*(\d+))?|(\d+)/(\d+) \[", re.I)
LEFT = re.compile(r"<(?:(\d+):)?(\d+):(\d+)")  # tqdm's time remaining, e.g. [12:01<1:03:22, 3.1it/s]
SPARK = "▁▂▃▄▅▆▇█"


def parse(path, loss_re):
    """[(step, loss)], [(step, grad_norm)], total steps if the log says, tqdm time left if any."""
    pts, grads, total, left, step = [], [], None, None, None
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
                g = GRAD.search(line)
                if g:
                    grads.append((step if step is not None else len(pts), float(g.group(1))))
    return pts, grads, total, left


def first_bad(pts, spike=3.0):
    """Index of the first NaN or blow-up (same rule as watch.py), or None."""
    h = []
    for i, (_, y) in enumerate(pts):
        if not math.isfinite(y) or (len(h) >= 10 and y > spike * statistics.median(h[-50:])):
            return i
        h.append(y)
    return None


def broke(run):
    """(step, what, fatal) for the first NaN, loss blow-up or grad-norm spike in a run, or None.

    Only loss problems are fatal (the curve is cut there). A grad-norm spike is marked but the
    curve carries on, since high-LR runs recover from one-step blips all the time.
    """
    _, pts, grads = run
    found = []
    c, g = first_bad(pts), first_bad(grads, 10.0)
    if c is not None:
        found.append((pts[c][0], "NaN" if not math.isfinite(pts[c][1]) else "blew up", True))
    if g is not None:
        found.append((grads[g][0], "grad norm", False))
    return min(found) if found else None


def cut_at(b):
    """Step past which the loss curve is meaningless, or None."""
    return b[0] if b and b[2] else None


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
    b = broke(runs[-1])
    c = cut_at(b)
    finite = [y for x, y in pts if c is None or x < c][-240:]  # healthy part only, a 1e6 blow-up would flatten it
    spark = ""
    if finite:
        k = max(1, len(finite) // 30)
        buckets = [statistics.mean(finite[i:i + k]) for i in range(0, len(finite), k)]
        lo, hi = min(buckets), max(buckets)
        spark = "".join(SPARK[int((b - lo) / (hi - lo or 1) * 7.999)] for b in buckets)
    tail = f"  min {min(finite):.4g}" if finite else ""
    if b is not None:
        tail += f" · {'✕' if b[2] else '⚠'} {b[1]} @ step {b[0]:,}"
    if len(runs) > 1:
        tail += f" · {len(runs) - 1} resume{'s' * (len(runs) > 2)}"
    return f"{head}\nloss {loss:.4g}  {spark}{tail}"


def svg(runs, total):
    W, H, L, R, T, B = 760, 340, 64, 24, 64, 46
    cut = [broke(r) for r in runs]
    ends = [cut_at(b) for b in cut]
    healthy = [y for (_, pts, _), e in zip(runs, ends) for x, y in pts if e is None or x < e]
    if not healthy:
        return None
    steps = [x for _, pts, _ in runs for x, _ in pts]
    xmin = min(steps)  # a resumed job starts mid-run; don't squash it against the right edge
    xmax = max([total or 0] + steps)
    xmax = xmax if xmax > xmin else xmin + 1
    lo, hi = min(healthy), max(healthy)
    hi = hi if hi > lo else lo + 1
    X = lambda x: L + (W - L - R) * (x - xmin) / (xmax - xmin)
    Y = lambda y: T + (H - T - B) * (1 - (min(y, hi) - lo) / (hi - lo)) if math.isfinite(y) else T - 4
    font = 'font-family="ui-monospace, Menlo, monospace"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" rx="12" fill="#191817"/>',
         f'<text x="{L}" y="34" {font} font-size="15" fill="#8b8783">loss · {escape(runs[0][0])}</text>']
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#2e2c2a"/>')
        o.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" {font} font-size="11" fill="#8b8783">{v:.3g}</text>')
    o.append(f'<text x="{L}" y="{H - 18}" {font} font-size="11" fill="#8b8783">{xmin:,}</text>')
    o.append(f'<text x="{W - R}" y="{H - 18}" text-anchor="end" {font} font-size="11" fill="#8b8783">step {xmax:,}</text>')
    for n, ((_, pts, _), b, e) in enumerate(zip(runs, cut, ends)):
        if not pts:
            continue
        last = n == len(runs) - 1
        if n:
            x0 = X(pts[0][0])
            o.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{T - 6}" y2="{H - B}" stroke="#d97757" stroke-dasharray="3 4" opacity=".7"/>')
            o.append(f'<text x="{x0 + 6:.1f}" y="{H - B - 8}" {font} font-size="11" fill="#d97757">resumed @ {pts[0][0]:,}</text>')
        shown = [p for p in pts if e is None or p[0] <= e]  # past a blow-up the curve is just the clip line
        k = max(1, len(shown) // 600)
        line = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in shown[::k] + shown[-1:])
        if last:
            # raw loss faded, EMA on top: real runs are too noisy to read otherwise
            o.append(f'<polyline points="{line}" fill="none" stroke="#d97757" stroke-width="1" opacity=".3"/>')
            a, ema, sm = 2 / (max(2, len(shown) // 40) + 1), None, []
            for x, y in shown:
                if math.isfinite(y):
                    ema = y if ema is None else a * y + (1 - a) * ema
                sm.append((x, ema if ema is not None else y))
            line = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in sm[::k] + sm[-1:])
        style = 'stroke="#d97757" stroke-width="2.5"' if last else 'stroke="#8b8783" stroke-width="2" stroke-dasharray="5 4"'
        o.append(f'<polyline points="{line}" fill="none" {style} stroke-linejoin="round"/>')
        if b is not None:
            col = "#ff6b6b" if b[2] else "#f2cc60"
            bx = X(b[0])
            anchor = "start" if bx < L + 90 else "end" if bx > W - R - 90 else "middle"  # keep the label inside the plot
            o.append(f'<text x="{bx:.1f}" y="{T - 10}" text-anchor="{anchor}" {font} font-size="12" font-weight="bold" fill="{col}">{"✕" if b[2] else "⚠"} {b[1]} @ {b[0]:,}</text>')
    step, loss = runs[-1][1][-1] if runs[-1][1] else (0, float("nan"))
    failed = ends[-1] is not None
    o.append(f'<text x="{W - R}" y="36" text-anchor="end" {font} font-size="22" font-weight="bold" '
             f'fill="{"#ff6b6b" if failed else "#e8e6e3"}">{"NaN" if not math.isfinite(loss) else f"{loss:.4g}"}</text>')
    if total:
        frac = min(1.0, step / total)
        o.append(f'<rect x="{L}" y="{H - 12}" width="{W - L - R}" height="4" rx="2" fill="#2e2c2a"/>')
        o.append(f'<rect x="{L}" y="{H - 12}" width="{(W - L - R) * frac:.1f}" height="4" rx="2" fill="{"#ff6b6b" if failed else "#7ee787"}"/>')
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
        pts, grads, t, left = parse(path, loss_re)
        runs.append((os.path.basename(path), pts, grads))
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
