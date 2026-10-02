---
name: babysit
description: >
  Watch a long-running ML training job overnight and step in when it breaks:
  NaN or exploding loss, loss spikes, CUDA OOM, crashes, hangs, SLURM jobs that
  die or time out, full disks. Shows progress bars, loss sparklines and loss
  charts. Waits without polling, diagnoses from the log, applies the
  fix the user allowed (usually resume from the last checkpoint with a safer
  setting), and leaves a morning report. Use when the user says "babysit",
  "watch my run", "keep an eye on training", "monitor this job", "I'm going to
  sleep", asks how a run is going, or starts a long training run and walks away. Works for local runs,
  SLURM jobs, and runs on a remote cluster over ssh.
---

# Babysit a training run

You are on night shift for someone's training run. They want to wake up to a
finished run or a clear note on what broke and what you did about it, not to a
dead job at step 4k.

Two scripts sit next to this file (the skill's base directory), both stdlib only:

- `watch.py` blocks until the run needs attention, prints one JSON event and
  exits. You never poll: start it in the background, end your turn, and you get
  woken when it exits.
- `chart.py` shows how the run is going: a progress bar and loss sparkline for
  the terminal, or an SVG loss chart that marks where it broke and resumed.

## 1. Set up (ask once, in one message, only for what you can't work out)

- **The run.** The log file it writes, plus either the local PID, the SLURM job
  id, or the ssh host it runs on. If they haven't started it yet, start it
  yourself with output to a log (`nohup ... > run.log 2>&1 & echo $!`, or
  `sbatch` and read the job id). Find the checkpoint dir and the resume flag
  in the training script now, while nothing is on fire.
- **What you may do when it breaks.** Default is *report only*. Ask if you may
  also *fix and resume*: restart from the last checkpoint with a safer setting.
  Never assume this permission.
- **How to reach them.** If a push-notification tool is available, use it for
  anything that needs a human. Otherwise the report file is the channel.

Then write the plan to `<log>.sleeptrain.md` next to the log: run, command,
checkpoint dir, permissions. You append every event and action there.

## 2. Watch

```bash
python3 <skill-dir>/watch.py --log runs/run.log --slurm 123456   # or --pid 4242
```

Run it with the Bash tool in the background, then end your turn. On a remote
cluster, keep the script local and pipe it over ssh (it is stdlib only):

```bash
ssh hpc 'python3 - --log ~/runs/123456.out --slurm 123456' < <skill-dir>/watch.py
```

ssh must not prompt. If the cluster needs 2FA, ask the user to open a
ControlMaster connection once before they go to sleep.

Tuning flags: `--spike 3` (loss above 3x the recent median), `--grad-spike 10`
(same for grad norm), `--stall 30` (minutes of silence), `--min-free 2` (GB of
disk), `--every 30` (poll seconds; keep it at 30+ on shared schedulers), and
`--loss-regex` if the log names the train loss something other than `loss` /
`train_loss` / `train/loss`.

The watcher exits with a `healthy` event every 20 minutes (`--max-minutes`), because
agent harnesses kill long background commands. That is the normal heartbeat: restart
it right away with `--from <offset>` and say nothing, unless they asked for
check-ins (then send the `chart.py` two-liner about once an hour). If a watcher
ever dies without an event, check the run yourself before restarting it.

## 3. When it wakes you

The event JSON has `event`, the offending `detail` line, `job_state`,
`last_loss`, `median_loss`, `last_grad_norm`, the last 40 log lines in `tail`, and an `offset`.
Read the tail before deciding anything. Then:

| event | first check | fix (only with fix-and-resume permission) |
|---|---|---|
| `spike` | Did it recover in the tail? Right after warmup or a data shard switch? | One spike that recovers: note it, keep watching. Repeated or growing: treat as `nan`. |
| `grad_spike` | The early warning: the loss often follows within a few steps. Is the grad norm still climbing in the tail? Compare the median grad norm and blip count over the last few thousand steps. | Climbing: act now as for `nan`, before the loss blows up. Blips that recover with no trend are this run's normal texture (common at high LR): restart with a higher `--grad-spike` (e.g. 20) so they stop waking you. |
| `nan` | Step it started, grad norm, LR at that point, fp16 vs bf16. | Resume from the last checkpoint *before* the blow-up with LR halved (or warmup doubled); for fp16, try bf16 or a lower loss scale. |
| `oom` | Which allocation, at what step: first step (config) or later (fragmentation, long batch)? | Halve the micro-batch and double grad accumulation so the global batch is unchanged; resume. |
| `crash` | The traceback. Code bug, data bug, or infrastructure (NCCL, node, disk full)? | Infrastructure: resume as is. Obvious one-line code or path bug: fix, note the diff, resume. Anything else: report. |
| `stall` | Is the process alive and using the GPU (`nvidia-smi`, `sstat -j`, `ps`)? Dataloader hang, NCCL timeout, full disk? | Hung but alive: kill and resume. Dead: treat as `crash`. |
| `disk` | What's filling it: checkpoints, logs, caches? | Never delete anything yourself. Report with the biggest directories (`du -sh`) and which old checkpoints could go. |
| `healthy` | Nothing: this is the 20-minute heartbeat. | Restart the watcher with `--from <offset>`. Check-in only if they asked for them. |
| `died` | SLURM state: `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED`, `FAILED`. | `TIMEOUT` / `NODE_FAIL` / `PREEMPTED`: resubmit with resume. Others: as `oom` / `crash`. |
| `exited` | The tail: did it finish cleanly or die? | As `done` or `crash`. |
| `done` | Final loss and eval numbers in the log. | Write the report. |

After a transient event you chose to ride out, restart the watcher with
`--from <offset>` so it skips the line you already handled. After a resume,
start a fresh watcher on the new log / PID / job id.

## 4. Show, don't just tell

Whenever you report on the run (when they ask "how's it going?", on a
heartbeat, after a resume, in the morning report), run `chart.py` with every log
of the run in order (`run.log run.r1.log ...`) and paste its two lines in a code
block:

```
run.r1.log  [████████████████░░░░░░░░]   66%  step 198/300  · ~1h 04m left
loss 1.707  ████▇▆▆▅▅▄▄▃▄▄▃▃▂▂▂▂▁▁▁▁▂▂▂▂▂  min 1.585 · 1 resume
```

For the morning report, and after every resume, also write the chart:
`chart.py run.log run.r1.log --svg <log>.chart.svg`, and embed it in the report
with `![loss](<log>.chart.svg)`. Pass `--total N` if the log doesn't print
`step N/TOTAL` or a tqdm bar. For a run on a remote cluster, `scp` the logs
into a temp dir and run `chart.py` locally.

## 5. Guardrails

- Never delete or overwrite checkpoints, logs or outputs. A resumed run writes
  to a new log file (`run.r1.log`, `run.r2.log`, ...).
- Change only what the table says: LR, warmup, micro-batch with matching grad
  accumulation, precision, resubmitting. Never touch the data, the model
  size, the eval or the total token budget. That's the user's experiment.
- At most 3 automatic resumes per night. After the third, report and stop.
  The same failure twice in a row means your fix didn't work, so report instead of
  trying a third variation.
- If a fix needs more GPUs, more money or a different queue, ask first.

## 6. Morning report

When the run ends, or you stop, finish `<log>.sleeptrain.md`. Keep it to
what they'll read in the first minute:

```markdown
## sleeptrain: run.log, 23:10 -> 06:42
![loss](run.log.chart.svg)

**Status:** finished, step 20000/20000, final loss 2.31
**Overnight:** NaN at step 8,410 (LR 3e-4, fp16). Resumed from step 8,000 at LR 1.5e-4 -> stable.
**Changed:** lr 3e-4 -> 1.5e-4 (run.r1.log). Nothing else.
**Look at:** loss spike at step 14,200 recovered on its own, probably a data shard switch.
```

Send the status line as a push notification if you can.
