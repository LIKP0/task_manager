#!/usr/bin/env python3
# Note: a pane inherits only PATH from tm; every other variable comes from the
# environment the tmux server first started in (measured on 3.2a). So python resolves
# correctly, but CONDA_PREFIX / LD_LIBRARY_PATH from `conda activate` are lost.
"""tm — queue up task lists and start them when a GPU frees up.

Every task runs in its own tmux session. tm does three things: scan the queue in
order, decide what can start, and watch rc files for results. It holds no
authoritative state of its own; everything is on disk next to tm.py (queue/, runs/),
so it can be killed and restarted at any time, and you can edit the queue by hand
while it is not running.

    tm                    consume task lists from queue/ (run it inside tmux)
    tm ls                 show progress; works whether or not tm is running
    tm add list.yaml      add to the queue (a copy; doing it by hand is the same)
    tm check list.yaml    parse without running, to see how it expands
    tm attach [name]      attach to a running task
    tm clean              remove tmux sessions left behind by failures
    tm hold / tm resume   pause and resume queue scanning, to edit the queue

tm's own settings live in tm_config.yaml, read once at startup. There are no
built-in defaults and no command-line override: to change them, stop tm, edit the
file, restart. Running tasks are unaffected.

Deliberately absent: automatic retries, automatically killing stuck tasks, and fair
resource scheduling. Scheduling is yours — queue order is priority, and tm follows it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tmlib import runner
from tmlib.config import (ConfigError, check_device_settings, load_plan,
                          unescape_braces)
from tmlib.gpu import GpuPool
from tmlib.settings import Settings, config_path, load_settings
from tmlib.store import LockBusy, Store, StoreError, now_stamp as store_now
from tmlib.view import Style, describe_wait, render


# --------------------------------------------------------------------------- #
# tm run — the main loop
# --------------------------------------------------------------------------- #

def cmd_run(store: Store, args, st: Style) -> int:
    try:
        store.check_writable()
    except StoreError as exc:
        # If rc files cannot be written, even successful tasks read as failures
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2

    try:
        with store.lock():
            return _loop(store, args, st)
    except LockBusy as exc:
        print(st.red(f"error: another tm is already running ({exc.holder})"), file=sys.stderr)
        print(st.dim("  see what it is doing: tm ls"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(st.yellow("\ntm: scheduling stopped. Running tasks are unaffected — "
                        "they are still going in their own tmux sessions."))
        print(st.dim("  tm ls for progress; starting tm again picks up where this left off."))
        return 130


def _loop(store: Store, args, st: Style) -> int:
    cfg: Settings = args.settings
    pool = GpuPool()
    waiting_since: dict[Path, float] = {}
    # path -> which problem we last reported for it, so each is announced once and
    # a *different* problem with the same file still gets through.
    complained: dict[Path, str] = {}

    print(st.bold(f"tm: watching {store.queue_dir}"))
    print(st.dim(f"    poll {cfg.poll:.0f}s · device_check "
                 f"{'on' if cfg.device_check else 'off'}  ({config_path(store.root)})"))
    idle_announced = False
    pause_announced = False

    while True:
        pool.refresh()
        # Read the active runs once. Each call parses every run.yaml under runs/, and
        # runs/ is never pruned, so this is the tick's dominant cost at any real
        # history size (~0.5s per call at 1000 runs).
        active = store.active()
        # The allocation table is rebuilt from disk every tick, so a restarted tm
        # never hands out a card someone else already holds. Claims are keyed by run
        # directory, not list name: duplicate list names are allowed, and releasing
        # by name would wipe the other claim on a shared card.
        pool.rebuild([(r.path.name, r.gpus, r.gpu_budget_gb, r.exclusive)
                      for r in active])

        for run in active:
            _advance(run, pool, st)

        paused = store.paused()
        if paused:
            # Stop the wait clock. It measures elapsed wall time, but nothing is
            # eligible to start while paused, so leaving it running would time out a
            # list for an hour it never had a chance to use.
            waiting_since.clear()
            if not pause_announced:
                print(st.yellow("tm: paused. Running tasks still advance; the queue is "
                                "not scanned. Resume with: tm resume"))
                pause_announced = True
        else:
            if pause_announced:
                print(st.green("tm: queue scanning resumed."))
                pause_announced = False
            _start_pending(store, pool, st, args, waiting_since, complained)

        # Re-read: _advance and _start_pending both change what is active.
        active, queued = store.active(), store.queued()
        if not active and not queued:
            # Nothing queued and nothing running: paused or not, there is nothing
            # left to wait for, so --once exits as usual
            if args.once:
                print(st.green("tm: queue empty, exiting."))
                return 0
            if not paused and not idle_announced:
                print(st.dim(f"tm: queue empty, standing by. Drop something into "
                             f"{store.queue_dir} and it starts automatically."))
                idle_announced = True
        else:
            idle_announced = False

        time.sleep(cfg.poll)


def _advance(run, pool: GpuPool, st: Style) -> None:
    """Advance one run: collect what finished, start what is next.

    Progress comes entirely from rc files (`run.scan()`), never from run.yaml. So as
    soon as the previous step writes rc=0, this call starts the next one, without
    waiting another tick.
    """
    run.reload()
    done, fail_i, fail_rc = run.scan()
    tasks = run.tasks

    if not tasks:
        # No run.yaml, or one with no tasks. `done >= len(tasks)` would be trivially
        # true and report a list that never ran a single step as successful — exactly
        # the false success the rc-file design exists to make impossible.
        run.set_state("broken", note="run.yaml is missing or lists no tasks")
        pool.release(run.gpus, run.path.name)
        print(st.red(f"\n<== {run.name} BROKEN — no run.yaml, or it lists no tasks"))
        print(st.dim(f"    {run.path}"))
        return

    if fail_i is not None:
        rec = tasks[fail_i - 1]
        run.set_state("failed", failed_step=fail_i, failed_rc=fail_rc)
        pool.release(run.gpus, run.path.name)
        print(st.red(f"\n<== {run.name} FAILED at step {fail_i}/{len(tasks)} "
                     f"({rec.name}) rc={fail_rc}"))
        _print_tail(rec, st)
        print(st.dim(f"    still there: tmux attach -t {rec.session}"))
        print(st.dim(f"    {run.path}"))
        return

    if done >= len(tasks):
        run.set_state("done")
        pool.release(run.gpus, run.path.name)
        print(st.green(f"\n<== {run.name} done ({len(tasks)}/{len(tasks)})"))
        return

    idx = done + 1
    rec = tasks[idx - 1]

    if not rec.started:
        sess = runner.Session(runner.session_name(run.name, run.run_id, idx, rec.name),
                              run.rc_path(idx))
        _launch(run, idx, rec, sess, pool, st)
        return

    # Ask about the session this step actually launched, recorded in run.yaml — not a
    # freshly computed name. Two runs of the same list name produce the same computed
    # name, so a recomputed one could report another run's live session as ours and
    # leave this one stuck in `running` forever instead of declaring it LOST.
    #
    # scan() has just confirmed step idx has no rc file, so only one question is
    # left: is the session alive? Check the session first, then re-check rc. The
    # wrapper writes rc before exiting, so if the task happened to finish between
    # the two reads the second one sees the rc and we do not misjudge it as LOST.
    if not runner.Session(rec.session).alive() and run.rc(idx) is None:
        # No rc file and no session: taken out by kill -9, the OOM killer or a
        # reboot. No evidence means failure; it must never continue as success.
        run.set_state("lost", failed_step=idx)
        pool.release(run.gpus, run.path.name)
        print(st.red(f"\n<== {run.name} LOST at step {idx}/{len(tasks)} ({rec.name})"))
        print(st.dim("    session gone with no exit code — hard-killed or the machine rebooted"))


def _print_tail(rec, st: Style, lines: int = 12) -> None:
    """Print the last lines of the pane on failure, so reading the error needs no attach."""
    tail = runner.Session(rec.session).capture(lines)
    if not tail:
        return
    print(st.dim(f"    --- last {len(tail)} lines of {rec.session} " + "-" * 30))
    for line in tail:
        print(st.dim("    | ") + line)


def _launch(run, idx: int, rec, sess: runner.Session, pool: GpuPool, st: Style) -> None:
    env = {"PYTHONUNBUFFERED": "1"}
    if run.gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in run.gpus)
    try:
        sess.launch(rec.cmd, Path(run.cwd), env, run.script_path(idx))
    except (runner.TmuxError, OSError) as exc:
        run.set_state("aborted", note=str(exc))
        # Terminal now, so release here rather than waiting for the next tick's
        # rebuild(): otherwise the cards stay booked for the rest of this queue scan
        # and lists below are refused a card nobody holds.
        pool.release(run.gpus, run.path.name)
        print(st.red(f"tm: cannot launch {run.name} step {idx}: {exc}"))
        return
    run.set_task(idx, session=sess.name, started=store_now())
    run.event(f"step {idx} {rec.name} -> {sess.name}")
    print(st.cyan(st.bold(f"\n==> {run.name} [{idx}] {rec.name}")) + st.dim(f"  {rec.cmd}"))
    print(st.dim(f"    tmux attach -t {sess.name}"))


def _claim(store: Store, path: Path, plan, gpus: list[int], st: Style,
           complained: dict[Path, str]):
    """claim() with the disk failure reported once. None means it stays queued.

    Every claim path goes through here: a claim touches the filesystem, and an
    unguarded one takes the whole scheduler down with it, stranding every other
    queued list over a problem with this one.
    """
    try:
        run = store.claim(path, plan, gpus)
    except OSError as exc:
        if complained.get(path) != "claim":
            complained[path] = "claim"
            print(st.red(f"tm: cannot start {plan.name}: {exc}"))
        return None
    complained.pop(path, None)
    return run


def _start_pending(store: Store, pool: GpuPool, st: Style, args,
                   waiting_since: dict[Path, float],
                   complained: dict[Path, str]) -> None:
    """Scan the queue in order and start whatever can start.

    Order is priority: a list that cannot fill its request reserves the free cards it
    could have used, so lists below it cannot take them this tick. Otherwise a list
    wanting two cards would never get to run.
    """
    now = time.monotonic()
    queued = store.queued()
    # Drop bookkeeping for lists that left the queue by hand. `rm` is the documented
    # way to cancel, and a stale entry here would otherwise hand its timeout clock to
    # whatever lands on the same filename next.
    # Both are the caller's objects, so mutate in place — rebinding would silently
    # leave _loop holding the unpruned originals.
    live = set(queued)
    for stale in [p for p in waiting_since if p not in live]:
        del waiting_since[stale]
    for stale in [p for p in complained if p not in live]:
        del complained[stale]

    for path in queued:
        try:
            plan = load_plan(path)
        except ConfigError as exc:
            if complained.get(path) != "parse":
                complained[path] = "parse"
                print(st.red(f"tm: cannot read {path.name}, skipping: {exc}"))
            continue
        if complained.get(path) == "parse":
            del complained[path]        # it parses now; a claim failure is separate

        gpus = pool.pick(plan.wait)
        if gpus is None:
            first = waiting_since.setdefault(path, now)
            if plan.wait.timeout is not None and now - first >= plan.wait.timeout:
                run = _claim(store, path, plan, [], st, complained)
                if run is None:
                    continue
                run.set_state("timeout")
                waiting_since.pop(path, None)
                print(st.yellow(f"tm: {plan.name} timed out waiting for a gpu "
                                f"({plan.wait.timeout:.0f}s), skipping"))
            continue

        if plan.wait.manages_gpu and args.settings.device_check:
            problems = check_device_settings(plan.tasks, plan.cwd, plan.wait.gpus)
            if problems:
                # Do not let it retry forever in the queue: move it out and explain
                run = _claim(store, path, plan, [], st, complained)
                if run is None:
                    continue
                run.set_state("aborted", note="device check failed")
                print(st.red(f"tm: {plan.name} failed the config device check, skipping:"))
                for msg in problems:
                    print(st.red(f"  - {msg}"))
                print(st.dim("  tm selects cards via CUDA_VISIBLE_DEVICES, so configs "
                             "must use relative indices."))
                continue

        run = _claim(store, path, plan, gpus, st, complained)
        if run is None:
            continue
        pool.allocate(gpus, run.path.name, plan.wait.gpu_free_gb or 0.0,
                      plan.wait.exclusive)
        waiting_since.pop(path, None)
        where = ",".join(f"gpu{g}" for g in gpus) or "no gpu"
        print(st.bold(f"\ntm: start {plan.name} on {where}  ({len(plan.tasks)} tasks)"))
        _advance(run, pool, st)


# --------------------------------------------------------------------------- #
# Other subcommands
# --------------------------------------------------------------------------- #

def cmd_ls(store: Store, args, st: Style) -> int:
    for line in render(store, st):
        print(line)
    return 0


def cmd_add(store: Store, args, st: Style) -> int:
    rc = 0
    for src in args.files:
        if not src.is_file():
            print(st.red(f"error: not found: {src}"), file=sys.stderr)
            rc = 2
            continue
        try:
            # Validate first, so bad yaml never reaches the queue. add and run see
            # the same file under the same rules, so passing here means passing there.
            load_plan(src)
        except ConfigError as exc:
            print(st.red(f"error: {src}: {exc}"), file=sys.stderr)
            rc = 2
            continue
        dst = store.add(src, seq=args.seq)
        print(f"queued  {st.cyan(dst.name)}")
        # Sequence numbers only order files that have them. A hand-copied `zzz.yaml`
        # contributes nothing to the numbering, so the new file can land ahead of it
        # and quietly jump the queue — the opposite of the documented FIFO priority.
        after = store.queued()
        if args.seq is None and after and after[-1] != dst:
            print(st.yellow(f"  warning: {dst.name} did not land last in the queue "
                            f"(after {after[-1].name})"))
            print(st.dim("  queue order is filename order; rename to fix it"))
    if rc == 0:
        print(st.dim("(order is filename order; rename to change it, rm to cancel)"))
    return rc


def cmd_check(store: Store, args, st: Style) -> int:
    try:
        plan = load_plan(args.file)
    except ConfigError as exc:
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2
    spec = plan.wait
    print(st.bold(f"{plan.name}: {len(plan.tasks)} tasks, cwd={plan.cwd}"))
    # Same renderer `tm ls` uses, so check cannot describe a list differently from
    # what you will see once it is queued.
    print(st.cyan("  [wait] ") + describe_wait(spec))
    for i, t in enumerate(plan.tasks, start=1):
        # Unescape `{{` / `}}` before display, or check would show something other
        # than what the shell receives. `{GPU}` is left alone: the card is only
        # decided at claim time and is unknown here.
        print(f"  {st.cyan(f'[{i}] {t.name}')}  {unescape_braces(t.cmd)}")
    if spec.manages_gpu:
        problems = check_device_settings(plan.tasks, plan.cwd, spec.gpus)
        for msg in problems:
            print(st.red(f"  ! {msg}"))
        if problems:
            return 2
    return 0


def cmd_attach(store: Store, args, st: Style) -> int:
    live = set(runner.list_sessions())
    targets = []
    for run in store.active():
        idx = run.current_index()
        tasks = run.tasks
        if idx <= len(tasks) and tasks[idx - 1].session in live:
            targets.append(tasks[idx - 1].session)
    if args.name:
        targets = [s for s in live if args.name in s] or [args.name]
    if not targets:
        print(st.yellow("No running tasks."), file=sys.stderr)
        return 1
    if len(targets) > 1:
        print(st.bold("Several are running; pick one:"))
        for name in targets:
            print(f"  tm attach {name}")
        return 1
    os.execvp("tmux", ["tmux", "attach", "-t", f"={targets[0]}"])
    return 0                                   # execvp does not return


def cmd_clean(store: Store, args, st: Style) -> int:
    """Remove panes left behind by failures. Running ones are never touched."""
    keep = set()
    for run in store.active():
        keep.update(t.session for t in run.tasks if t.session)
    victims = [s for s in runner.tm_sessions() if s not in keep]
    if not victims:
        print(st.dim("No sessions to clean up."))
        return 0
    for name in victims:
        if not args.yes:
            print(f"  would kill  {name}")
        else:
            runner.Session(name).kill()
            print(f"  killed  {name}")
    if not args.yes:
        print(st.dim(f"add -y to actually do it ({len(victims)} session(s))"))
    return 0


def cmd_hold(store: Store, args, st: Style) -> int:
    """Hold the queue so you can reorder it safely.

    This blocks only the starting of new lists. Running tasks are untouched; they are
    already in their own tmux sessions and no longer involve the scheduler. It works
    while tm is not running too — tm will come up paused.
    """
    store.pause()
    print(st.yellow("tm: queue scanning paused. Running tasks are unaffected."))
    print(st.dim(f"  queue directory {store.queue_dir}    resume with: tm resume"))
    return 0


def cmd_resume(store: Store, args, st: Style) -> int:
    if not store.resume():
        print(st.dim("tm: not currently paused."))
        return 0
    print(st.green("tm: queue scanning resumed."))
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tm", description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The queue is queue/ in the repo directory and its order is the "
               "priority; copying a yaml in by hand is the same as tm add. tm's own "
               "settings live in tm_config.yaml and are read once at startup.")
    p.add_argument("--root", type=Path, default=None,
                   help="override the state directory (default: the repo); "
                        "TM_ROOT does the same")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="consume the queue (default)")
    # Mode switches for this invocation only. Settings all live in tm_config.yaml.
    run.add_argument("--once", action="store_true",
                     help="exit when the queue drains instead of standing by")

    sub.add_parser("ls", help="show progress (works when tm is not running)")

    add = sub.add_parser("add", help="add a yaml to the queue")
    add.add_argument("files", nargs="+", type=Path)
    add.add_argument("--seq", type=int, default=None,
                     help="sequence number; appended to the end by default")

    check = sub.add_parser("check", help="parse without running")
    check.add_argument("file", type=Path)

    attach = sub.add_parser("attach", help="attach to a running task")
    attach.add_argument("name", nargs="?", default=None)

    clean = sub.add_parser("clean", help="remove tmux sessions left by failures")
    clean.add_argument("-y", "--yes", action="store_true", help="actually do it")

    sub.add_parser("hold", help="pause: start nothing new, so you can edit the queue")
    sub.add_parser("resume", help="undo hold")

    args = p.parse_args(argv)
    st = Style()
    store = Store(args.root)
    # Every command creates queue/ and runs/ first. Creating them lazily means a repo
    # that has never run tm has no queue directory to look at, while the README and
    # tm ls both tell you to copy a yaml into it. Failures are not reported here;
    # cmd_run's check_writable() gives a more accurate message.
    try:
        store.ensure()
    except OSError:
        pass

    # tm_config.yaml is read once for every subcommand. A missing or malformed file
    # fails here, so every command hits it; there is no config error that shows up
    # only on one path ("tm ls is fine but tm run will not start").
    try:
        args.settings = load_settings(store.root)
    except ConfigError as exc:
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2

    handlers = {None: cmd_run, "run": cmd_run, "ls": cmd_ls, "add": cmd_add,
                "check": cmd_check, "attach": cmd_attach, "clean": cmd_clean,
                "hold": cmd_hold, "resume": cmd_resume}
    if args.cmd is None and not hasattr(args, "once"):
        args.once = False           # bare `tm` means `tm run`

    return handlers[args.cmd](store, args, st)


if __name__ == "__main__":
    sys.exit(main())
