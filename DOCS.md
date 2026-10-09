# tm — technical reference

How tm behaves and why: every command, every key, every rule. For installing and a
first run, see the [README](README.md).

Queue up task lists and start them when a GPU frees up. Every task runs in its own
tmux session.

A task runs only if the previous one exited 0. On failure the list stops there and
the pane is kept, so you can attach and look around.

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
  paused                      present while `tm hold` is in effect
  queue/010_ccfm_c.yaml       here = not started. Copying one in == tm add
  runs/20260808_143301_ccfm_c/
                              here = in progress
    list.yaml                 the original, moved out of queue/
    run.yaml                  tm's plan: state, cards held, session and final
                              command for each step
    01.sh                     the wrapper script tmux actually executes
    01.rc                     *the exit code the task wrote* — the only evidence
    events.log                one line per claim, step start and state change
  archive/20260807_090000_ccfm_b/
                              here = finished; same contents, never touched again
```

The central split is **tm writes the plan, the task writes the result**. rc files
keep appearing after tm dies, and tm has no say in what they contain.

Where a list sits says where it is in its life: `queue/` → `runs/` → `archive/`. tm
moves a run into `archive/` as soon as it records the final state, so the scheduler
reads only `runs/` each tick and costs the same however long the history grows. A
list skipped before it ever starts (timed out, or refused by the device check) goes
straight from `queue/` to `archive/`. A finished run still in `runs/` was finished
by a tm that died before moving it; the next tick moves it. The history only takes
disk space; `tm prune` clears it when you want.

## Commands

```
tm [--root DIR] <subcommand>
```

| Subcommand | What it does |
|---|---|
| `tm` / `tm run` | Consume the queue. Stays resident and stands by when empty |
| `tm ls [-a]` | Progress: running, queued, the newest 10 finished runs (`-a` for all), and used VRAM and utilisation per card |
| `tm add F...` | Add yaml files to the queue. Each is parsed and capacity-checked first; a bad one is refused and never reaches the queue |
| `tm check F` | Parse without running: how variables expand, how it starts, plus the device and capacity checks. Exit 2 if any fails |
| `tm attach [name]` | Attach to the running step. With a name, any tmux session containing it; lists them if several match |
| `tm clean [-y]` | Remove tmux sessions left by failures. Lists them unless given `-y` |
| `tm prune [-d N] [-n N] [-y]` | Delete finished runs from `archive/` past either limit (default 30 days / newest 30). Lists them unless given `-y` |
| `tm hold` / `tm resume` | Pause and resume **queue scanning**, so you can edit it |

| Option | Applies to | What it does |
|---|---|---|
| `--root DIR` | global | Override the state directory (default: the repo). `TM_ROOT` does the same. Mostly test scaffolding — see below |
| `--once` | `run` | Exit when the queue drains instead of standing by |
| `--seq N` | `add` | Sequence number, 0–999; appended to the end by default. Past 999 the filename would no longer sort into place, so `tm add` refuses and asks you to renumber the queue |
| `-d/--days N` | `prune` | Remove runs that finished more than N days ago (default 30) |
| `-n/--keep N` | `prune` | Remove all but the newest N finished runs (default 30) |

The command line carries **mode switches only, never settings**. Settings live in
`tm_config.yaml`, and there is no way to override the contents of a task list from the
command line either.

`--root` exists for testing, not for daily use. All disk access goes through one
object, so pointing it elsewhere gives a test a throwaway state tree instead of
dirtying the repo's own `queue/` and `runs/`. The one real-world case is a repo on a
read-only or network filesystem that cannot hold the state itself. It is **not** a
way to run two queues at once: the lock is per root, so two tm processes with
different roots each hold their own, see the same physical cards, and hand the same
GPU to both.

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

wait:                    # this or now: (exactly one): hold until enough GPU is free
  gpu_free_gb: 50        # wait for a card with >= 50 GiB free
  gpus: 1                # how many cards
  gpu_index: any         # any | 0 | [0, 1]
  stable_for: 120        # the condition must hold for 120 seconds

cwd: ~/my_project       # required; working directory for every command

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
| `name` | **Required.** Appears in `tm ls`, the run directory name and tmux session names (`tm-<name>-<run id>-01-<task>`, where the run id is the run's start date and time). Never derived from the filename |
| `vars` | Values for `{KEY}`, each a single value (no lists or mappings). A value is spliced into the command and then treated like the rest of it: `{GPU}` expands and `{{ }}` is a literal brace there too. No command-line override; fix them here before running |
| `wait` | Queue until a card is free, below. **Exactly one of `wait` and `now` is required** |
| `now` | Start at once on named cards, or `now: cpu`; see [Starting now](#starting-now) |

`name` used to be derived from the filename, which also meant stripping the `010_`
prefix that `tm add` adds — one implicit transform existing only to undo another,
whose result ended up in tmux session names. Now it is what you wrote, and anything
invalid is rejected immediately. Queue filenames carry ordering only.

Each task is `{name, cmd}`; a bare string works too and is named `step1`, `step2`.
**Names may use letters, digits, `_` and `-` only**, because they become part of a
tmux session name and `.` and `:` are tmux target separators.

The run id in a session name is that run's start date and time. It is what lets you
re-queue a list whose previous run failed: the failed pane stays pinned, and without a
run id the retry would collide with it and be aborted before running a step.

An undefined `{KEY}` is an error rather than a silent pass-through, so a typo in a
path never costs you hours. Write `{{` / `}}` for literal braces (python f-strings,
awk's `{{print $1}}`). `{GPU}` is special: it expands after a card is acquired, to
the **physical index** (`0,1` for two cards), which makes it useful for naming output
directories. Commands are expanded once, at claim time, and run.yaml stores the exact
line handed to tmux; list.yaml is never re-read after that.

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
| `gpu_free_gb` | **required** | Require at least this much free VRAM (**GiB**; nvidia-smi's MiB / 1024) |
| `gpus` | `1` | How many cards |
| `gpu_index` | `any` | Restrict the candidates: `any` / `0` / `[0, 1]`. Must list at least `gpus` of them |
| `stable_for` | `120` | **Seconds**, max 1800. How long the condition must hold continuously. The cap is the sample history tm keeps; a larger window could never be satisfied, so it is rejected rather than accepted and never met |
| `timeout` | none | **Seconds** (`7200` = two hours). Give up on this list after waiting this long, mark it `timeout` and move on. **Omitting it means waiting for ever, not refusing to wait**, and that is the right default: a timeout does not keep the list queued, it takes the yaml out of `queue/` having run nothing, so a card that frees up on day four finds nothing to run. The clock is frozen while `tm hold` is in effect, but it lives only in tm's memory, so restarting tm restarts the count |
| `exclusive` | `true` | No second tm run may share the card. `false` allows sharing, below |

`gpu_free_gb` is required: it is what the list waits for. Leaving out the whole
`wait:` block used to mean "no GPU, start at once" — the same spelling for a CPU job
and for a GPU job you had pinned to a card by hand, so tm could tell neither apart.
Both are now said out loud with `now:`.

`stable_for` is not optional padding. A job that just started is still reading data
and has not built its memory pool, so nvidia-smi shows the card as empty; move in
then and both sides OOM thirty seconds later. Requiring the condition to hold
**continuously** for two minutes closes almost all of that window, and the only cost
is two minutes.

Stability is judged from a history of VRAM samples, one per tick (`poll`), kept for
`1800 + 60` seconds — just over the largest `stable_for` accepted. The history lives
only in memory, so a restarted tm re-establishes stability from scratch, which is the
right behaviour anyway.

`gpu_free_gb` is **per card**, so `gpus: 2` with `gpu_free_gb: 40` asks for two cards
with 40 GiB free each, not 80 GiB in total.

`tm check` and `tm add` measure the request against the machine and refuse one no
card can ever satisfy — `gpu_free_gb` larger than any card, or more `gpus` than exist:

```
error: big.yaml: gpu_free_gb: 200.0 is per card and needs 1 card(s) that big,
       but 0 qualify (gpu0: 95.6 GiB, gpu1: 95.6 GiB) — the list can never start
```

Without it such a list queues cleanly and then waits for ever, indistinguishable in
`tm ls` from one waiting its turn. This is the only rule in `tm add` that depends on
the machine rather than the file, so it is skipped where nvidia-smi is absent rather
than refusing a yaml written for a different host. Only the totals are read; a busy
card is irrelevant. A list copied into `queue/` by hand skips both commands and so
skips this check too.

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

### Starting now

`now:` skips the queue. Use it to run something next to whatever tm is already
running — a quick baseline beside a training job, or a CPU job:

```yaml
now:                     # on the cards named, as soon as tm sees the list
  gpu_index: 1           # required: 1 | [0, 1]; the length is the card count
  gpu_free_gb: 20        # required: VRAM each card must have free right now

now: cpu                 # no card at all
```

**Only VRAM can stop a `now:` list.** Exclusive claims, `stable_for`, queue order and
the cards reserved by lists above it are agreements between waiting lists, and `now:`
is you overriding them. The two VRAM tests from
[Exclusive and shared](#exclusive-and-shared) still apply, measured once, this tick:

| Test | Criterion |
|---|---|
| Measured free | nvidia-smi free >= `gpu_free_gb` |
| Declared budget | card total - sum of budgets of tm runs on it >= `gpu_free_gb` |

The second test is what keeps you off a tm run that has just started and still looks
empty. If either test fails on any named card, the list is **refused, not queued**:
it goes to `archive/` as `ABORT` with the numbers in run.yaml's `note`, and tm prints them.
Waiting would make it a queued list that ignores exclusive and jumps the queue, which
is not what "now" means. Fix the card or the number and add it again.

Once started it is a run like any other: `CUDA_VISIBLE_DEVICES` and `{GPU}` are set to
its cards, the device check applies, and its claim on the card is **exclusive**, so
waiting lists keep off. `tm ls` shows it as `now:gpu1`.

`now: cpu` launches with `CUDA_VISIBLE_DEVICES` set empty, so a task that reaches for
cuda fails at once instead of quietly landing on gpu0. `{GPU}` in such a list is
rejected at parse time; there is no card for it to name.

`now:` takes no other key. `wait:` and `now:` together, or neither, is an error.

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

A file copied in by hand without a number (`zzz.yaml`) still sorts by name, so a
later `tm add` can land ahead of it. `tm add` warns when the file it just queued did
not land last.

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
# tm: tm-ccfm_c-20260808-143301-01-train
# Everything tm hands to tmux. Run it directly to reproduce this step.
set -o pipefail
cd /home/me/my_project || exit 1

(
python train.py --config config/xxx.yaml
)
rc=$?

echo $rc > .../01.rc.tmp && mv .../01.rc.tmp .../01.rc

[ $rc -eq 0 ] && exit 0
exec bash -i
```

tmux itself is handed one line, `bash <path>/01.sh`, which every login shell (fish,
csh, zsh) runs the same way, so the user's default shell never has to be POSIX.

Points worth knowing:

- **`set -o pipefail`**: a POSIX pipeline's exit status is that of its last command,
  so in `python train.py | tee log` a crashed train still yields 0. pipefail gets the
  real value. To ignore a failure deliberately, write your own `|| true`. Without
  bash, tm falls back to `sh` and drops pipefail.
- **The `( )` subshell**: if the command calls `exit` (or ends in `exec`), running it
  unwrapped would take the wrapper with it and the rc file would never be written.
  The parentheses sit on lines of their own so that a trailing `# comment` in the
  command cannot comment out the `)`.
- **`.tmp` then `mv`**: rename is atomic, so the rc file is either absent or complete.
- **Succeed and vanish, fail and stay pinned**: successful sessions exit and leave no
  junk; failed panes keep their full scrollback, and attaching gives an interactive
  shell in the job's cwd and environment (conda is live), so you can debug in place.
- To reproduce a step, run `bash NN.sh`. Once the run is in `archive/` the command
  runs as before, but the script still names its rc file under `runs/`, so that one
  write fails — a rerun cannot overwrite the evidence of the original run.

### The task's environment

tm injects two environment variables: `PYTHONUNBUFFERED=1`, and `CUDA_VISIBLE_DEVICES`
set to the run's cards — empty for `now: cpu`.

Everything else comes from tmux, not from tm. Measured on tmux 3.2a: the pane gets
tm's `PATH`, but every other variable comes from the environment the **tmux server**
was first started in. So `python` resolves to the interpreter of the conda env tm was
started from and training runs, but `CONDA_PREFIX`, `CONDA_DEFAULT_ENV` and
`LD_LIBRARY_PATH` set by `conda activate` are lost if the server was started outside
that env. Jobs that only need the interpreter are unaffected. Jobs that depend on
those variables (self-built CUDA extensions, scripts reading `CONDA_PREFIX`) should
set them in the command, or start the tmux server from the activated env.

The pane kept open after a failure is a fresh `bash -i`, which reads `~/.bashrc`, so
conda is live there when you attach.

## When something fails

The failing pane is kept, and tm prints its last 12 lines, so reading the error takes
no attach:

```
<== ccfm_c FAILED at step 2/3 (test) rc=1
    --- last 12 lines of tm-ccfm_c-20260808-143301-02-test ---------------
    | Traceback (most recent call last):
    | FileNotFoundError: no such checkpoint: ...
    still there: tmux attach -t tm-ccfm_c-20260808-143301-02-test
    /home/me/task_manager/archive/20260808_143301_ccfm_c
```

Later steps do not run. `tm ls` shows the list as `FAILED  ccfm_c  1/3`. When you are
done looking, `tm clean -y` removes the leftover sessions; running ones are never
touched.

Every run ends in one of six states, shown in the `RECENT` column:

| State | Meaning |
|---|---|
| `ok` | Every step exited 0 |
| `FAILED` | A step exited non-zero; the rest were skipped. The line gives `step<N> <name> rc=<code>` |
| `LOST` | The session vanished with no exit code — **treated as a failure**, since no evidence means failure |
| `TIMEOUT` | `wait.timeout` expired before a card was free; it never started |
| `ABORT` | Refused before starting (the `trainer.devices` check, or a `now:` list whose cards lack the VRAM), or the session could not be launched |
| `BROKEN` | The run directory has no readable `run.yaml`. It is never reported as done — a run that executed nothing must not look successful |

`tm ls` looks roughly like this:

```
tm: pid 31337 since 2026-08-08 14:20:11

RUNNING
  list    gpu   step         time
  ccfm_c  gpu1  [2/3] test  1h04m  -> tmux attach -t tm-ccfm_c-20260808-143301-02-test

QUEUED   (order = priority; rename to change it)
  file               tasks    start
  020_ddpm_fdg.yaml  2 tasks  1x50GiB on any gpu
  030_broken.yaml             BAD: .../queue/030_broken.yaml: missing 'tasks:'

RECENT
  state   list   steps  took  finished
  ok      tok      3/3    3s  08-08 23:49
  FAILED  tpipe    0/2    1s  08-08 23:49  step1 piped rc=3
  ... 14 more, tm ls -a for all

GPUS
  gpu0:   42.8/95.6 GiB used   util 100%
  gpu1:    0.7/95.6 GiB used   util   0%
```

The first line says `not running` when no tm holds the lock (it tries the flock
rather than trusting the pid written in the file, which outlives a hard-killed tm),
and adds `[queue paused · tm resume]` during a hold. A running step whose session has
already vanished shows `session gone` until the next tick marks it LOST. In the
`gpu` column, `now:gpu1` is a `now:` run and `cpu` a `now: cpu` one. The GPU
figures are coloured red when a card has under 5 GiB free and yellow above 50%
utilisation.

### Stuck jobs are flagged, never killed

A pane with no output for 30 minutes is marked `⚠ silent 42m10s` in yellow. tm does
nothing else about it: saving a checkpoint, the gap between epochs and CPU-bound eval
are all long silences, and an automatic kill would eventually hit a healthy job while
it was writing a checkpoint — precisely the worst moment to interrupt. Whether to kill
it is your call.

### When tm itself hits trouble

One bad list or one bad directory never takes the scheduler down with it; everything
else keeps running. Each problem is reported once, not every tick.

| Problem | What tm does |
|---|---|
| A queued yaml does not parse | Skipped and shown as `BAD:` in `tm ls`; picked up automatically once fixed |
| A claim fails on disk | Reported; the list is left for the next tick. If the yaml already left `queue/`, the run directory without a run.yaml turns into `BROKEN` next tick — loud rather than silent |
| tmux cannot launch a step | The run is marked `ABORT` and its cards released |
| A run directory becomes unwritable | That run is set aside. **Its cards stay reserved**: tm cannot tell whether the task is still on them, and handing them out would be the one unrecoverable mistake. Fix the directory and restart tm |
| A finished run cannot be moved to `archive/` | It stays in `runs/`, costing one run.yaml read per tick; a restarted tm tries again |
| The state directory is not writable at startup | `error: ... is not writable`, exit 2, before anything starts |

## Configs must use relative device indices

After picking cards, tm sets `CUDA_VISIBLE_DEVICES=<physical index>`, so the child
process sees only `gpus` cards numbered from 0. A config saying `trainer.devices: [1]`
then fails to find its card, and so does `test.py --cuda 1`.

So **tm checks this before starting** — rather than letting you find out six hours
later:

- every `--config <path>` in each command (several per step are all checked): if the
  yaml has `trainer.devices`, it must be `[0, ..., gpus-1]`, `gpus`, `-1` or `auto`.
  The last two mean "every visible card", which is exactly the cards tm assigned. A
  config without `trainer.devices`, or one that cannot be read, is left alone.
- every `--cuda N`: N must be below `gpus`.

```
tm: ddpm_fdg failed the config device check, skipping:
  - task 'train': /home/me/my_project/config/ddpm_fdg.yaml
      trainer.devices is [1], must be [0] (or 1, -1, auto) — tm assigns the physical card via CUDA_VISIBLE_DEVICES
```

The check applies to every list with a card (`wait:`, or `now:` with `gpu_index`),
since their cards are remapped through `CUDA_VISIBLE_DEVICES`. It runs in `tm check`, and again when the list reaches the
front of the queue, where a failing list is marked `ABORT` and moved to `archive/`
instead of retrying for ever. `tm add` does not run it.

Use `{GPU}` when you want the physical index (for logs or output directory names). The
check knows only these two conventions (Lightning-style `trainer.devices` and a
`--cuda` flag); if your configs have a different shape and it false-alarms, set
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
