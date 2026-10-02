"""Render assets/demo.gif frame by frame (SVG -> PNG via rsvg-convert -> GIF via magick)."""
import os
import subprocess
from html import escape

W, H, LH, PAD, TOP, MAXL = 960, 620, 25, 28, 64, 22
FONT = "JetBrainsMonoNL Nerd Font Mono, Menlo, monospace"
C = dict(bg="#191817", bar="#262422", fg="#e8e6e3", dim="#8b8783", claude="#d97757",
         red="#ff6b6b", yellow="#f2cc60", green="#7ee787", blue="#79c0ff")

def seg(text, color="fg", bold=False):
    return (text, color, bold)

lines, frames = [], []

def snap(ms, clock):
    frames.append((list(lines), ms, clock))

def add(*segs, ms=450, clock="23:58"):
    lines.append(list(segs))
    snap(ms, clock)

add(seg("$ ", "green", True), seg("python3 train.py --lr 3e-3 > run.log 2>&1 &"), ms=700)
add(seg("[1] 4242", "dim"), ms=500)
add(seg(""), ms=50)
prompt = "babysit run.log, you can fix and resume. going to bed"
lines.append([])
for i in range(8, len(prompt) + 8, 8):
    lines[-1] = [seg("> ", "claude", True), seg(prompt[:i])]
    snap(90, "23:58")
snap(700, "23:58")
add(seg(""), ms=50)
add(seg("● ", "claude"), seg("Watching run.log (pid 4242), checkpoints every 50 steps."), ms=500)
add(seg("  Going quiet until something breaks. Sleep well."), ms=500)
add(seg("  └ watch.py in background · 0 tokens while the run is healthy", "dim"), ms=1600)
add(seg("                    z z z . . .", "blue"), ms=1300, clock="00:47")
snap(900, "02:13")
for step, loss, g, col in [(117, "2.0069", "0.76", "dim"), (118, "1.9423", "2.27", "fg"),
                           (119, "1.9758", "6.80", "yellow"), (120, "1.9366", "20.41", "red")]:
    add(seg(f"step {step}/300 | loss {loss} | "), seg(f"grad_norm {g}", col, col == "red"), seg(" | lr 0.003", "dim"),
        ms=420, clock="02:13")
add(seg("⚠ grad_spike: grad_norm 20.41 at step 120 (median 0.80)", "red", True), ms=1300, clock="02:13")
add(seg(""), ms=50, clock="02:13")
add(seg("● ", "claude"), seg("Grad norm went 0.8 → 20 in 3 steps. The loss hasn't moved yet,"), ms=600, clock="02:14")
add(seg("  but it will. Killing it and resuming from ckpts/step_100.json"), ms=600, clock="02:14")
add(seg("  at lr 1.5e-3 → run.r1.log"), ms=1300, clock="02:14")
add(seg("                    z z z . . .", "blue"), ms=1300, clock="04:30")
add(seg("run.r1.log  ", "fg", True), seg("[████████████████████████]", "green"), seg("  100%  step 300/300"), ms=500, clock="06:42")
add(seg("loss 1.223  ", "fg", True), seg("████▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁", "claude"), seg("  min 1.198 · 1 resume", "dim"), ms=800, clock="06:42")
add(seg("● ", "claude"), seg("Done. Morning report + loss chart → run.log.sleeptrain.md"), ms=4000, clock="06:42")

def svg(shown, clock):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" rx="14" fill="{C["bg"]}"/>',
           f'<path d="M0 14a14 14 0 0 1 14-14h{W-28}a14 14 0 0 1 14 14v26H0z" fill="{C["bar"]}"/>']
    for i, col in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        out.append(f'<circle cx="{24 + i * 22}" cy="20" r="6.5" fill="{col}"/>')
    out.append(f'<text x="{W/2}" y="25" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{C["dim"]}">claude · sleeptrain</text>')
    out.append(f'<text x="{W-PAD}" y="25" text-anchor="end" font-family="{FONT}" font-size="14" font-weight="bold" fill="{C["blue"]}">☾ {clock}</text>')
    for i, line in enumerate(shown[-MAXL:]):
        y = TOP + i * LH
        spans = "".join(
            f'<tspan fill="{C[c]}" font-weight="{"bold" if b else "normal"}">{escape(t)}</tspan>' for t, c, b in line)
        out.append(f'<text x="{PAD}" y="{y}" font-family="{FONT}" font-size="17" xml:space="preserve">{spans}</text>')
    out.append("</svg>")
    return "\n".join(out)


here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "demo.gif")
here = os.path.join(here, ".frames")  # scratch frames, gitignored
os.makedirs(f"{here}/frames", exist_ok=True)
args = ["magick"]
for n, (shown, ms, clock) in enumerate(frames):
    base = f"{here}/frames/{n:03d}"
    open(base + ".svg", "w").write(svg(shown, clock))
    subprocess.run(["rsvg-convert", base + ".svg", "-o", base + ".png"], check=True)
    args += ["-delay", str(max(2, ms // 10)), base + ".png"]
subprocess.run(args + ["-loop", "0", "-layers", "Optimize", out], check=True)
print(len(frames), "frames ->", out, os.path.getsize(out) // 1024, "KB")
