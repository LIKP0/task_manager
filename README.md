# tm

A GPU task queue for one machine. Put your train → test → eval steps in a yaml list,
queue it, and tm starts it when a card frees up. Each step runs in its own tmux
session, and the next one runs only if the previous exited 0.

```
$ tm ls
tm: pid 31337 since 2026-08-08 14:20:11

RUNNING
  ccfm_c         gpu1     [2/3] test           1h04m  -> tmux attach -t tm-ccfm_c-20260808-143301-02-test

QUEUED   (order = priority; rename to change it)
  020_ddpm_fdg.yaml        2 tasks   1x50GiB on any gpu

RECENT
  ok      unet_base      3/3    5h12m  08-08 13:02
  FAILED  unet_wide      1/3    2m00s  08-08 07:49  step2 test rc=1

GPUS
  gpu0:   42.8/95.6 GiB used   util 100%
  gpu1:   61.3/95.6 GiB used   util  98%
```

- **Waits for a stable card.** A list starts only once a card has had enough free VRAM
  for a while, so it never walks into a job that is still loading.
- **Exit codes decide.** A step that crashes, is OOM-killed or vanishes in a reboot
  counts as failed, and the steps after it do not run.
- **Restart any time.** tm keeps no state in memory; kill it, start it again, and it
  carries on. `tm ls` works even when tm is not running.
- **Failed panes stay.** Attach to a failed step and debug it where it died.

Built for one machine, one user, NVIDIA cards. No cluster scheduling, no fair sharing,
no automatic retries.

## Install

```bash
git clone https://github.com/LIKP0/task_manager ~/task_manager
pip install pyyaml
ln -s ~/task_manager/tm.py ~/.local/bin/tm
```

Also needs `tmux`, `bash` and `nvidia-smi`. Tested with Python 3.12.7, PyYAML 6.0.3,
tmux 3.2a, bash 5.1.16 and NVIDIA driver 595.84 on Ubuntu 22.04. The queue and run
records live inside the clone, so put it somewhere writable.

## Quickstart

Write a list, say `lists/unet.yaml` (`lists/` is gitignored;
`task_list.example.yaml` is the annotated template):

```yaml
name: unet_base
cwd: ~/my_project               # every command runs here

wait:                           # omit to start right away, without a GPU
  gpu_free_gb: 40               # one card with 40 GiB free
  stable_for: 120               # for two minutes straight

vars:
  CFG: configs/unet_base.yaml

tasks:
  - name: train
    cmd: python train.py --config {CFG}
  - name: test
    cmd: python test.py --config {CFG} --ckpt "$(ls -t ckpt/*.ckpt | head -1)"
  - name: eval
    cmd: python eval.py --config {CFG}
```

```bash
tm check lists/unet.yaml   # show what it will run and wait for
tm add lists/unet.yaml     # queue it
tm                         # run the scheduler; leave it in a tmux window
tm ls                      # check progress from anywhere
```

Two things to know before the first run:

- tm sets `CUDA_VISIBLE_DEVICES`, so the chosen card is always `0` inside the job.
  Write `devices: [0]` or `auto` in configs, never a physical index; tm refuses lists
  that get this wrong.
- Start tmux and tm from the conda env your jobs need.

## Everyday use

| To | Do |
|---|---|
| Queue a list | `tm add F.yaml`, or copy it into `queue/` |
| Reprioritise | Rename files in `queue/`; filename order is run order |
| Cancel | `rm` it from `queue/` |
| Edit the queue | `tm hold`, edit, `tm resume` |
| Watch a step | `tm attach` |
| Clear failed panes | `tm clean -y` |
| Trim history | `tm prune -y` (keeps the newest 30, none older than 30 days) |
| Restart tm | Kill it and run `tm`; running steps carry on |

`clean` and `prune` only list what they would remove until given `-y`.

## Using tm from a coding agent

tm suits coding agents (Claude Code, Codex, ...) that run long experiments for you.
Teach the agent once instead of every session:

1. Ask it to read this repo (`README.md`, `DOCS.md`, `task_list.example.yaml`) and
   write itself a skill for tm: where lists go, how to write one for your project,
   `tm check` before `tm add`, `tm ls` for progress.
2. Invoke the skill by hand, or add a rule to the agent's instructions (`CLAUDE.md`,
   `AGENTS.md`, ...) so it uses tm on its own for long or multi-stage jobs:

```
Long training runs and multi-stage pipelines (train → test → eval) go through tm:
write a task list, tm check it, tm add it. Quick smoke tests run directly.
```

## Configuration

| To change | Edit |
|---|---|
| Poll interval, the device check | `tm_config.yaml` (`poll`, `device_check`); restart tm to apply |
| When a list may start | Its `wait:` block: `gpu_free_gb`, `gpus`, `gpu_index`, `stable_for`, `timeout`, `exclusive` |
| History kept | `tm prune -d DAYS -n COUNT` |
| Silence warning (30 min), finished runs shown (10) | `SILENT_WARN`, `RECENT_SHOWN` in `tmlib/view.py` |

`tm_config.yaml` has no defaults in code: every key must be there. The device check
understands Lightning-style `trainer.devices` and a `--cuda` flag; turn it off if your
configs look different.

## How it works

Each step runs from a wrapper script that writes its exit code to a file; that file
is the only thing tm trusts, and no file plus no session means the step was lost.
A list moves through `queue/` → `runs/` → `archive/` as it waits, runs and finishes.
Cards are held for the whole list, and the queue is scanned top-down so a list
needing two cards is not starved by ones needing one.

[`DOCS.md`](DOCS.md) has the full reference: every command and key, the scheduling
rules, and the reasoning behind them.

## License

MIT
