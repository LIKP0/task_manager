"""Human-facing output.

Everything `tm ls` prints is assembled by reading disk and tmux on the spot, so it
works whether or not tm is running. That is what keeping state on disk buys.
"""

from __future__ import annotations

import sys
from datetime import datetime

from . import runner
from .config import ConfigError, load_plan
from .gpu import query_gpus
from .store import TIME_FMT, Store

# How long a pane must be silent before it is flagged. Checkpointing and CPU-bound
# eval stay quiet for a long time, so do not set this low.
SILENT_WARN = 30 * 60.0

# Colour thresholds for the GPU table. Red once a card is too full to be useful to
# anyone, yellow while someone is clearly computing on it.
LOW_FREE_GIB = 5.0
BUSY_UTIL_PCT = 50


class Style:
    def __init__(self, enabled: bool | None = None):
        self.enabled = sys.stdout.isatty() if enabled is None else enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, t: str) -> str:   return self(t, "1")
    def dim(self, t: str) -> str:    return self(t, "2")
    def green(self, t: str) -> str:  return self(t, "32")
    def red(self, t: str) -> str:    return self(t, "31")
    def yellow(self, t: str) -> str: return self(t, "33")
    def cyan(self, t: str) -> str:   return self(t, "36")


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _elapsed(stamp: str, until: str = "") -> str:
    if not stamp:
        return ""
    try:
        t0 = datetime.strptime(stamp, TIME_FMT)
        t1 = datetime.strptime(until, TIME_FMT) if until else datetime.now()
    except ValueError:
        return ""
    return fmt_duration((t1 - t0).total_seconds())


def describe_wait(spec, short: bool = False) -> str:
    """One sentence describing what a list is waiting for.

    Shared by `tm ls` and `tm check`: check exists to preview what ls will show, so
    the two must not drift. `short` is the compact form for the queue table.
    """
    if not spec.manages_gpu:
        return "no gpu needed" if short else "no gpu requirement — starts immediately"
    where = ("gpu " + ",".join(map(str, spec.gpu_index))
             if spec.gpu_index is not None else "any gpu")
    if short:
        return (f"{spec.gpus}x{spec.gpu_free_gb:.0f}GiB on {where}"
                f"{'' if spec.exclusive else ', shared'}")
    return (f"{spec.gpus} x {spec.gpu_free_gb:.1f} GiB on {where}, "
            f"stable for {spec.stable_for:.0f}s, "
            f"{'exclusive' if spec.exclusive else 'shared'}")


def describe_gpus(indices: list[int], empty: str) -> str:
    return ",".join(f"gpu{g}" for g in indices) or empty


def render(store: Store, st: Style) -> list[str]:
    """The complete output of `tm ls`."""
    out: list[str] = []
    sessions = set(runner.list_sessions())
    # One pass over runs/, partitioned here. Calling active() and runs() separately
    # parsed every run.yaml twice, and runs/ is never pruned.
    everything = store.runs()
    active = [r for r in everything if not r.done]
    finished = [r for r in everything if r.done]

    holder = store.lock_holder()
    line = st.bold("tm: ") + (st.green(holder) if holder else st.dim("not running"))
    if store.paused():
        line += st.yellow("   [queue paused · tm resume]")
    out.append(line)

    # ---- running ---------------------------------------------------------------
    if active:
        out.append("")
        out.append(st.bold("RUNNING"))
    for run in active:
        done, _, _ = run.scan()
        tasks = run.tasks
        idx = done + 1
        gpu = describe_gpus(run.gpus, "-")
        step = tasks[idx - 1].name if idx <= len(tasks) else "-"
        line = (f"  {run.name:<14} {gpu:<8} [{idx}/{len(tasks)}] {step:<12} "
                f"{_elapsed(run.started):>7}")

        if idx <= len(tasks) and tasks[idx - 1].session:
            sess = runner.Session(tasks[idx - 1].session)
            if sess.name in sessions:
                silent = sess.silent_for()
                if silent is not None and silent >= SILENT_WARN:
                    line += st.yellow(f"  ⚠ silent {fmt_duration(silent)}")
                line += st.dim(f"  -> tmux attach -t {sess.name}")
            else:
                line += st.red("  session gone")
        out.append(line)

    # ---- queued ----------------------------------------------------------------
    queued = store.queued()
    if queued:
        out.append("")
        out.append(st.bold("QUEUED") + st.dim("   (order = priority; rename to change it)"))
    for path in queued:
        try:
            plan = load_plan(path)
            note = describe_wait(plan.wait, short=True)
            out.append(f"  {path.name:<24} {len(plan.tasks)} tasks   {st.dim(note)}")
        except ConfigError as exc:
            out.append(f"  {path.name:<24} {st.red('BAD: ' + str(exc).splitlines()[0])}")

    # ---- finished --------------------------------------------------------------
    # Every finished run, newest first — no truncation. runs/ is the whole history,
    # and a cut-off list quietly hides the run you were looking for.
    if finished:
        out.append("")
        out.append(st.bold("RECENT"))
    for run in finished:
        done, fail_i, fail_rc = run.scan()
        tasks = run.tasks
        mark = {"done": st.green("ok    "), "failed": st.red("FAILED"),
                "lost": st.red("LOST  "), "timeout": st.yellow("TIMOUT"),
                "aborted": st.yellow("ABORT ")}.get(run.state, run.state[:6].ljust(6))
        line = (f"  {mark}  {run.name:<14} {done}/{len(tasks)}  "
                f"{_elapsed(run.started, run.finished):>7}  {st.dim(run.finished[5:16])}")
        if fail_i and fail_i <= len(tasks):
            rec = tasks[fail_i - 1]
            line += f"  {st.red(f'step{fail_i} {rec.name} rc={fail_rc}')}"
            if rec.session in sessions:
                line += st.dim(f"  -> tmux attach -t {rec.session}")
        out.append(line)

    # ---- gpus ------------------------------------------------------------------
    try:
        gpus = query_gpus()
        out.append("")
        out.append(st.bold("GPUS"))
        for g in gpus:
            bar = (st.red if g.free_gib < LOW_FREE_GIB else
                   st.yellow if g.util > BUSY_UTIL_PCT else st.green)
            out.append(f"  gpu{g.index}: {bar(f'{g.free_gib:6.1f}')}/{g.total_gib:.1f} GiB free"
                       f"   util {g.util:3d}%")
    except ConfigError as exc:
        out.append("")
        out.append(st.yellow(f"GPUS: {exc}"))

    if not active and not queued:
        out.append("")
        out.append(st.dim(f"Queue is empty. Drop a yaml in: cp list.yaml {store.queue_dir}/"))
    return out
