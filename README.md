# tm

Queue up task lists and start them when a GPU frees up. Every task runs in its own
tmux session.

A task runs only if the previous one exited 0. On failure the list stops there and
the pane is kept, so you can attach and look around.

Requires `pyyaml` and `tmux`, plus `nvidia-smi` if you use `wait.gpu_free_gb`.

```bash
ln -s ~/task_manager/tm.py ~/.local/bin/tm    # install once

tm add lists/ccfm_c.yaml     # add to the queue (a copy into queue/)
tm                           # start the scheduler (run it inside tmux)
tm ls                        # progress; works whether or not tm is running
```

## How it is organised

tm holds no authoritative state. The queue is a directory, progress is the exit-code
files the tasks write themselves, and all of it lives in the repo directory next to
`tm.py` rather than in `~/.tm`, where it would be harder to browse. So:

- tm can be killed and restarted at any time without losing progress
- `tm ls` is a separate process and works when tm is not running
- you can edit the queue by hand while tm is stopped

```
~/task_manager/
  tm_config.yaml              tm's own settings (tracked in git)
                              everything below is gitignored: runtime state is not code
  lock                        single-instance mutex (flock, released on death)
  queue/010_ccfm_c.yaml       here = not started. Copying one in == tm add
  runs/20260808_143301_ccfm_c/
    list.yaml                 the original, moved out of queue/
    run.yaml                  tm's plan: state, cards held, session and final
                              command for each step
    01.sh                     the wrapper script tmux actually executes
    01.rc                     *the exit code the task wrote* — the only evidence
    events.log
```

The central split is **tm writes the plan, the task writes the result**. rc files
keep appearing after tm dies, and tm has no say in what they contain.

## Commands

```
tm [--root DIR] <subcommand>
```

| Subcommand | What it does |
|---|---|
| `tm` / `tm run` | Consume the queue. Stays resident and stands by when empty |
| `tm ls` | Progress: running, queued, every finished run, and free VRAM per card |
| `tm add F...` | Add yaml files to the queue (validated first; bad ones are refused) |
| `tm check F` | Parse without running: see how variables expand and what it will wait for |
| `tm attach [name]` | Attach to a running task; lists them if there are several |
| `tm clean [-y]` | Remove tmux sessions left by failures. Lists them unless given `-y` |
| `tm hold` / `tm resume` | Pause and resume **queue scanning**, so you can edit it |

| Option | Applies to | What it does |
|---|---|---|
| `--root DIR` | global | Override the state directory (default: the repo). `TM_ROOT` does the same |
| `--once` | `run` | Exit when the queue drains instead of standing by |
| `--seq N` | `add` | Sequence number; appended to the end by default |

The command line carries **mode switches only, never settings**. Settings live in
`tm_config.yaml`, and there is no way to override the contents of a task list from the
command line either.

## tm's settings: `tm_config.yaml`

| Key | Meaning |
|---|---|
| `poll` | Scheduler poll interval in seconds. Each tick samples VRAM, advances running runs and scans the queue |
| `device_check` | Check `trainer.devices` in task configs before starting (see below) |

**There are no defaults in the code.** Every key must be present in the file, which is
why it is tracked in git and ships with the tool. A default living in Python would put
the real value in two places, and reading the config would then tell you only what was
overridden rather than what tm will do.

**Read once at startup. No command-line override, no re-reading while running.** To
change something, stop tm, edit the file, start it again. Running tasks live in their
own tmux sessions, are not part of the scheduler, and survive the restart untouched.
The cost is about zero, and in return the file always describes the running tm,
instead of having to reconstruct which flags it was started with.

A missing file, a missing key or a misspelled one (`pol1: 60`) is an error, never a
silent fallback. Silence would leave you believing a setting applied when it did not.

If you run with a custom `--root` or `TM_ROOT`, that directory needs its own copy.

## What a task list looks like

```yaml
name: fused_tsample      # required; used in the run directory and session names

wait:                    # optional: hold until enough GPU is free.
  gpu_free_gb: 50        # wait for a card with >= 50 GiB free
  gpus: 1                # how many cards
  gpu_index: any         # any | 0 | [0, 1]
  stable_for: 120        # the condition must hold for 120 seconds

cwd: ~/my_project  # required; working directory for every command

vars:                    # {KEY} placeholders
  CFG: config/ccfm_2p5d_fused_gauss_tsample.yaml

tasks:
  - name: train
    cmd: python train.py --config {CFG}
  - name: test
    cmd: python test.py --config {CFG} --run multiout_fused --cuda 0
```

| Top-level key | Meaning |
|---|---|
| `tasks` | **Required**, executed in order |
| `cwd` | **Required.** The pane's starting directory, and so the base for every relative path in your commands (scripts, configs, output directories). No default: `tm add` copies the list into `queue/`, so anything relative to the yaml itself would drift |
| `name` | **Required.** Appears in `tm ls`, the run directory name and tmux session names (`tm-<name>-01-<task>`). Never derived from the filename |
| `vars` | Values for `{KEY}`. No command-line override; fix them here before running |
| `wait` | Start conditions, below |

`name` used to be derived from the filename, which also meant stripping the `010_`
prefix that `tm add` adds — one implicit transform existing only to undo another,
whose result ended up in tmux session names. Now it is what you wrote, and anything
invalid is rejected immediately. Queue filenames carry ordering only.

Each task is `{name, cmd}`; a bare string works too and is named `step1`, `step2`.
**Names may use letters, digits, `_` and `-` only**, because they become part of a
tmux session name and `.` and `:` are tmux target separators.

An undefined `{KEY}` is an error rather than a silent pass-through, so a typo in a
path never costs you hours. Write `{{` / `}}` for literal braces (python f-strings,
awk's `{{print $1}}`). `{GPU}` is special: it expands after a card is acquired, to
the **physical index**, which makes it useful for naming output directories.

Commands go to bash, so `&&`, pipes, `$(...)` and environment prefixes all work.
`$(...)` is evaluated **when that step actually starts**, so it can pick up a
checkpoint the previous step just wrote:

```yaml
- name: test
  cmd: python test.py --config {CFG} --ckpt "$(ls -t {CKPT_DIR}/*.ckpt | head -1)"
```

### Waiting for a GPU

| Key | Default | Meaning |
|---|---|---|
| `gpu_free_gb` | none | Require at least this much free VRAM (**GiB**; nvidia-smi's MiB / 1024). Omit to ignore GPUs entirely |
| `gpus` | `1` | How many cards |
| `gpu_index` | `any` | Restrict the candidates: `any` / `0` / `[0, 1]`. Must list at least `gpus` of them |
| `stable_for` | `120` | **Seconds.** How long the condition must hold continuously |
| `timeout` | none | **Seconds** (`7200` = two hours). Give up on this list after waiting this long, mark it `timeout` and move on. The timer lives only in tm's memory, so restarting tm restarts the count |
| `exclusive` | `true` | No second tm run may share the card. `false` allows sharing, below |

`stable_for` is not optional padding. A job that just started is still reading data
and has not built its memory pool, so nvidia-smi shows the card as empty; move in
then and both sides OOM thirty seconds later. Requiring the condition to hold
**continuously** for two minutes closes almost all of that window, and the only cost
is two minutes.

Do not use `on:` as a key. YAML 1.1 parses bare `on`/`off`/`yes`/`no` as booleans —
the well-known GitHub Actions trap — which is why the key for choosing cards is
called `gpu_index`.

### Exclusive and shared

**Exclusive by default**: a card with a tm run on it leaves the candidate pool
entirely, no matter how much VRAM is spare. This is structural, not a function of
`gpu_free_gb`.

A list with `exclusive: false` may share a card, and then two independent tests both
have to pass:

| Test | Keeps out | Criterion |
|---|---|---|
| Measured free | **other people's** processes | nvidia-smi free >= `gpu_free_gb` for `stable_for` seconds |
| Declared budget | **tm's own** runs | card total - sum of budgets declared on it >= `gpu_free_gb` |

The second is not optional. Right after tm starts A, A is still importing torch and
nvidia-smi shows the card as empty; going by the measured value alone lets B in, and
both OOM once their memory pools are built. How much A intends to use is something tm
already knows — it is A's own `gpu_free_gb` — and **known facts should not be guessed
at by sampling**.

So in shared mode `gpu_free_gb` means two things at once: "I need this much free" and
"I intend to use this much". Fill in your peak.

Pairing takes the conservative side: **if either party asks for exclusive, the card is
exclusive.** If A says `exclusive: true`, B cannot join even with `false` — otherwise
A's declaration would mean nothing.

## Scheduling

**Queue order is priority**, which is filename order. `tm add` numbers files
automatically (`010_`, `020_`, ...), so changing priority is renaming and cancelling
is `rm`. tm rescans the directory every tick, so **online and offline edits take the
same path** — there is no privileged channel.

Each tick scans from the top of the queue:

```
reserved = set()
for each pending list (filename order):
    available = stably free cards - allocated - reserved
    enough     -> take them all at once and start
    not enough -> put the free cards it could have used into reserved, so lists
                  below cannot touch them this tick
```

- **Atomic acquisition removes deadlock**: a list either holds everything it needs
  (running, waiting for nothing) or holds nothing (waiting, blocking nothing). There
  is no hold-and-wait, so there is no cycle.
- **reserved removes starvation**: a list wanting two cards is not picked apart by a
  later list wanting one.
- The cost is cards idling while the bigger list assembles its set. That
  predictability is bought on purpose.

Lanes — card 0 running one series, card 1 another — are not a separate concept; they
fall out of `gpu_index`. Two lists with `gpu_index: 0` serialise, two with different
indices run in parallel.

**Only one tm can schedule on a machine** (flock). A second one exits with an error
rather than competing for cards.

### Editing the queue by hand

`queue/` is an ordinary directory: change priority with `mv`, cancel with `rm`,
disable but keep with a rename to `.yaml.off` (`queued()` filters by suffix, so
anything that is not `.yaml`/`.yml` is invisible). All of these are **atomic** — a
tick sees either the old state or the new one, never something in between.

**Only one thing has an in-between state: writing a file in place.** `cat >
010_x.yaml`, `echo >`, or any slow-writing script leaves the file half-written for a
moment. If the truncation lands on a task boundary the result is still valid YAML,
just missing steps — and tm will run that truncated list without complaint. So there
are exactly two correct ways to put something in the queue:

```bash
tm add list.yaml                        # writes .tmp then renames
cp list.yaml queue/015_x.yaml.tmp && mv queue/015_x.yaml{.tmp,}
```

To edit several files without anything being picked up mid-edit, use `tm hold`:

```bash
tm hold          # stop scanning the queue; running tasks continue as normal
...              # reorder, delete, edit
tm resume
```

`hold` blocks exactly one thing: **starting new lists from the queue**. It works
whether or not tm is running, since it is just the presence of a `paused` file.

## How a step is judged finished

tm is not the task's parent process. The bash inside the tmux pane is what actually
`wait()`s for the exit code, and it writes that code to `NN.rc` — the only channel
between them. That gives three states:

| Session | rc file | Verdict |
|---|---|---|
| alive | absent | still running |
| — | present | finished; the exit code is the file contents |
| **gone** | **absent** | **LOST — died without reporting; treated as a failure** |

That last row is what holds the design up. Without it, a train taken out by `kill -9`,
the OOM killer or a reboot reads as success, and test runs against half a checkpoint.

The design guarantees one direction only: **it may report success as failure, but
never failure as success.** Cannot write to disk -> no rc -> failure. Hard-killed ->
no rc -> failure. That is also why tm verifies the state directory is writable before
starting anything, rather than discovering it halfway through.

### The wrapper script

What actually executes for each step is written to `NN.sh` in the run directory:

```bash
#!/usr/bin/bash
# tm: tm-ccfm_c-01-train
set -o pipefail
cd /home/me/my_project || exit 1

( python train.py --config config/xxx.yaml )
rc=$?

echo $rc > .../01.rc.tmp && mv .../01.rc.tmp .../01.rc

[ $rc -eq 0 ] && exit 0
exec bash -i
```

Points worth knowing:

- **`set -o pipefail`**: a POSIX pipeline's exit status is that of its last command,
  so in `python train.py | tee log` a crashed train still yields 0. pipefail gets the
  real value. To ignore a failure deliberately, write your own `|| true`.
- **The `( )` subshell**: if the command calls `exit` (or ends in `exec`), running it
  unwrapped would take the wrapper with it and the rc file would never be written.
- **`.tmp` then `mv`**: rename is atomic, so the rc file is either absent or complete.
- **Succeed and vanish, fail and stay pinned**: successful sessions exit and leave no
  junk; failed panes keep their full scrollback, and attaching gives an interactive
  shell in the job's cwd and environment (conda is live), so you can debug in place.
- To reproduce a step, run `bash NN.sh`.

tm injects two environment variables: `PYTHONUNBUFFERED=1`, and `CUDA_VISIBLE_DEVICES`
when a card was acquired.

## When something fails

The failing pane is kept, and tm prints its last 12 lines, so reading the error takes
no attach:

```
<== ccfm_c FAILED at step 2/3 (test) rc=1
    --- last 12 lines of tm-ccfm_c-02-test ------------------------------
    | Traceback (most recent call last):
    | FileNotFoundError: no such checkpoint: ...
    still there: tmux attach -t tm-ccfm_c-02-test
    /home/me/task_manager/runs/20260808_143301_ccfm_c
```

Later steps do not run. `tm ls` shows the list as `FAILED  ccfm_c  1/3`. When you are
done looking, `tm clean -y` removes the leftover sessions; running ones are never
touched.

`tm ls` looks roughly like this:

```
tm: pid 31337 since 2026-08-08 14:20:11

RUNNING
  ccfm_c         gpu1     [2/3] test          1h04m   -> tmux attach -t tm-ccfm_c-02-test

QUEUED   (order = priority; rename to change it)
  020_ddpm_fdg.yaml        2 tasks   1x50GiB on any gpu

RECENT
  ok      tok            3/3       3s  08-08 23:49
  FAILED  tpipe          0/2       1s  08-08 23:49  step1 piped rc=3

GPUS
  gpu0:   52.8/95.6 GiB free   util 100%
  gpu1:   94.9/95.6 GiB free   util   0%
```

### Stuck jobs are flagged, never killed

A pane with no output for 30 minutes is marked `⚠ silent 42m10s` in yellow. tm does
nothing else about it: saving a checkpoint, the gap between epochs and CPU-bound eval
are all long silences, and an automatic kill would eventually hit a healthy job while
it was writing a checkpoint — precisely the worst moment to interrupt. Whether to kill
it is your call.

## Configs must use relative device indices

After picking cards, tm sets `CUDA_VISIBLE_DEVICES=<physical index>`, so the child
process sees only `gpus` cards numbered from 0. A config saying `trainer.devices: [1]`
then fails to find its card, and so does `test.py --cuda 1`.

So **tm checks this before starting**: it extracts `--config <path>` from each
command, reads that yaml, and refuses to start the list if `trainer.devices` is wrong
— rather than letting you find out six hours later.

```
tm: ddpm_fdg failed the config device check, skipping:
  - task 'train': /home/me/my_project/config/ddpm_fdg.yaml
      trainer.devices is [1], must be [0] (or 1) — tm assigns the physical card via CUDA_VISIBLE_DEVICES
```

Use `{GPU}` when you want the physical index (for logs or output directory names). If
your configs have a different shape and the check only false-alarms, set
`device_check: false` in `tm_config.yaml`.

## Logs

**tm does not manage logs.** Each task runs in its own tmux session, so its output is
in that pane's scrollback: `tmux attach` or `tm attach` to read it, with progress bars
and colour intact. To keep a copy, add `| tee train.log` to the command yourself
(pipefail is already on, so a crashed train is not swallowed by tee).

This is deliberate: training scripts already write their own logs, and a second copy
from tm would be redundant.

## What it does not do

- **No automatic retries** (see below).
- **No automatically killing stuck tasks** (see above).
- **No guarantee of semantic correctness between tasks.** tm reads exit codes only. It
  does not know whether train wrote a checkpoint, or whether test read that
  checkpoint. That is yours to get right when writing the list.
- **No fair resource scheduling.** Scheduling is yours; queue order is priority.
- **No runtime overrides of any kind.** There is no `-v KEY=VALUE`: values for `vars:`
  live in the yaml, and settings live in `tm_config.yaml`, read once at startup. Edit the
  files rather than patching things on the command line. (The `-v` that used to exist
  was global, applying to every list in the queue, while `tm run` is a long-lived
  process and lists are queued at arbitrary times — "the `-v` tm started with" and
  "this list" never lined up in time.)

## Roadmap

Neither of these is built, and neither is urgent. They are listed so the reasoning
does not have to be reconstructed next time.

- **Automatic retries.** There are none today: a failed step stops and waits for you.
  Contention for cards is already handled *beforehand* by `stable_for`, and avoiding
  the problem beats retrying — a retry presupposes you already wasted one startup.
  Genuine bugs, wrong configs and NaNs all fail again on a second run. A supervisor's
  value is not in getting back up but in **not doing the wrong thing**: when train
  crashes, do not run test against half a checkpoint.

  If it is ever built, only *external* failures are worth retrying, so the first
  problem is telling them apart from internal ones — say, retrying only on a
  whitelist of exit codes, or only when the process died before producing anything,
  with an attempt limit and backoff. An undiscriminating `max_retries: 3` (an earlier
  version had one; it was removed) should not come back: it will faithfully run a
  doomed config three times while holding the card.

- **Parallel tasks within one list.** Execution is strictly serial today. The only
  case that makes sense is shardable preprocessing, since train -> test -> eval is
  inherently sequential. Doing it means settling the failure semantics (does one
  failed shard fail the step?) and how VRAM is divided.
