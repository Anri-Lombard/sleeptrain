# sleeptrain

**Claude watches your training run so you can sleep.**

You start a 10-hour run at midnight. At 02:13 the loss goes NaN, and you find out at 08:00.
sleeptrain is a Claude Code plugin that sits on that run overnight. When it breaks, Claude reads
the log, works out what happened, resumes from the last good checkpoint with a safer setting
(if you allowed it), and leaves you a report for the morning.

<p align="center"><img src="assets/explainer.gif" width="760" alt="Animated chart of an overnight demo run: the grad norm starts climbing at step 133 while the loss still looks fine, Claude says it will resume from step 100 because step 150 is already corrupted, the loss explodes on the abandoned run, the resumed run trains cleanly to step 300, and by morning the report says finished, final loss 1.418, one resume."></p>

In the morning you get a report with a loss chart that marks where the run broke and where Claude
picked it up:

<p align="center"><img src="assets/chart.svg" width="680" alt="Loss chart: the first run in grey stops where the grad norm exploded at step 120, a dashed line marks the resume from step 100, and the resumed run in orange trains down to 1.22."></p>

```
## sleeptrain: run.log, 23:10 -> 06:42
Status:    finished, step 20000/20000, final loss 2.31
Overnight: NaN at step 8,410 (LR 3e-4, fp16). Resumed from step 8,000 at LR 1.5e-4 -> stable.
Changed:   lr 3e-4 -> 1.5e-4 (run.r1.log). Nothing else.
Look at:   loss spike at step 14,200 recovered on its own, probably a data shard switch.
```

<details>
<summary>The unedited Claude Code session behind that animation (7× speed)</summary>
<p align="center"><img src="assets/session.gif" width="760" alt="A real Claude Code session, sped up 7 times: asked to babysit a training run, Claude loads the sleeptrain skill, starts the watcher, notices the grad norm climbing, skips the corrupted step 150 checkpoint, resumes from step 100 at half the learning rate, and reports the finished run."></p>
</details>

## Install

```
/plugin marketplace add Anri-Lombard/sleeptrain
/plugin install sleeptrain@sleeptrain
```

## Use

Tell Claude what to watch:

> babysit my run: logs in runs/run.log, slurm job 123456. You can resume from checkpoint if it breaks.

or just `/sleeptrain:babysit`.

Running a sweep? One watcher covers every SLURM job you own, including jobs that start later:

> babysit all my jobs on hpc tonight, the metrics are in runs/*/train_log.jsonl Claude asks for anything it can't work out, starts the watcher and
goes quiet until something happens.

Try it in 30 seconds with the fake run that diverges:

```bash
python3 examples/flaky_train.py --lr 3e-3 > run.log 2>&1 &
```

> babysit run.log, pid is $!, you may fix and resume

## What it catches

| Event | Detected from |
|---|---|
| NaN / inf loss | the train loss in your log: `loss 2.31`, `loss=nan`, `{'loss': 2.31}`, `train/loss: ...` |
| Loss spike | loss above 3x the median of the last 50 steps |
| Exploding gradients | grad norm NaN or 10x its recent median, usually a few steps *before* the loss blows up |
| CUDA OOM | `out of memory`, `OutOfMemoryError`, `oom-kill`, `Killed` |
| Crash | Python tracebacks, CUDA / NCCL errors, segfaults |
| Hang | no log output for 30 minutes while the job is running |
| Full disk | under 2 GB free where the run writes (checkpoints pile up) |
| Dead job | local PID gone, or SLURM state `TIMEOUT`, `FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED`; in sweep mode, for every job you own |

Works with anything that prints a loss: PyTorch, HF Trainer, Lightning, nanoGPT, torchtune, JAX.

## Ask how it's going

> how's my run?

```
run.r1.log  [████████████████░░░░░░░░]   66%  step 198/300  · ~1h 04m left
loss 1.707  ████▇▆▆▅▅▄▄▃▄▄▃▃▂▂▂▂▁▁▁▁▂▂▂▂▂  min 1.585 · 1 resume
```

Add "check in every hour" and Claude sends you that every hour too (as a push notification if
you have them set up).

## How it works

`skills/babysit/watch.py` is a ~260-line, stdlib-only Python script that follows the log and
blocks until something needs attention, then prints one JSON event and exits. Claude runs it
in the background, so it costs **almost no tokens while the run is healthy**: a quiet check-in
every 20 minutes (agent background tasks get killed if they run longer) and a real wake-up only
when something breaks. Since it's stdlib only, it also runs on a cluster login node over ssh with nothing
installed:

```bash
ssh hpc 'python3 - --log runs/123456.out --slurm 123456' < watch.py
```

`skills/babysit/chart.py` draws the progress bar, sparkline and SVG chart, also stdlib only.

The skill (`skills/babysit/SKILL.md`) is the playbook: what to check for each event, which fixes
are allowed (LR, warmup, micro-batch with matching grad accumulation, precision, resubmitting),
and the guardrails.

## Guardrails

- **Report-only by default.** Claude only resumes runs if you said it could.
- Never deletes or overwrites checkpoints or logs. Resumed runs write to `run.r1.log`, `run.r2.log`, ...
- Never touches your data, model size, eval or token budget.
- At most 3 automatic resumes per night, and it stops if the same fix fails twice.

## Things to know

- Claude Code has to stay running overnight. Keep the laptop awake (`caffeinate -i` on macOS)
  or run Claude on a machine that doesn't sleep.
- Use an interactive session. Headless `claude -p` exits when Claude ends its turn, so nothing is
  left to wake when the watcher fires.
- On clusters with 2FA, open an ssh ControlMaster connection before bed so the watcher can
  reconnect without a prompt.
- The loss regex covers common log formats. If yours differs, pass `--loss-regex`.

<p align="center"><img src="assets/meme.png" width="420" alt="Meme. Top: an alarm clock at 03:00, Setting a 3am alarm to check if the loss went NaN. Bottom: a moon and a laptop with a recovered loss curve, Letting Claude sleep-train your model."></p>

## License

MIT
