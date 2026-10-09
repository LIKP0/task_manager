"""Human-facing output.

Everything `tm ls` prints is assembled by reading disk and tmux on the spot, so it
works whether or not tm is running. That is what keeping state on disk buys.
"""

from __future__ import annotations

import sys
from datetime import datetime

from . import runner
from .config import ConfigError, NowSpec, load_plan
from .gpu import query_gpus
from .store import TIME_FMT, Store

# How long a pane must be silent before it is flagged. Checkpointing and CPU-bound
# eval stay quiet for a long time, so do not set this low.
SILENT_WARN = 30 * 60.0

# Colour thresholds for the GPU table. Red once a card is too full to be useful to
# anyone, yellow while someone is clearly computing on it.
LOW_FREE_GIB = 5.0
BUSY_UTIL_PCT = 50

# Finished runs `tm ls` shows unless given -a. Failures get no exemption: the cap is
# about how long the listing is, and the line below it says how many were left out.
RECENT_SHOWN = 10


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


def describe_start(spec, short: bool = False) -> str:
    """One sentence describing how a list starts: what it waits for, or that it doesn't.

    Shared by `tm ls` and `tm check`: check exists to preview what ls will show, so
    the two must not drift. `short` is the compact form for the queue table.
    """
    if isinstance(spec, NowSpec):
        if not spec.gpus:
            return "now, cpu" if short else "cpu only — no card visible"
        where = "gpu " + ",".join(map(str, spec.gpu_index))
        if short:
            return f"now on {where}, {spec.gpu_free_gb:.0f}GiB"
        return (f"on {where}, {spec.gpu_free_gb:.1f} GiB free per card or it is "
                f"refused — skips the queue, stable_for and exclusive claims")
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


def _columns(rows: list[list[str]], right: frozenset[int] = frozenset()) -> list[list[str]]:
    """Pad every cell to the widest in its column, so columns line up however long
    the names are. Plain text only: escape codes would count as width, so style a
    cell after padding it."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [[cell.rjust(w) if i in right else cell.ljust(w)
             for i, (cell, w) in enumerate(zip(row, widths))] for row in rows]


# Finished states as `tm ls` spells them, and their colour
_STATE_MARK = {"done": ("ok", "green"), "failed": ("FAILED", "red"),
               "lost": ("LOST", "red"), "timeout": ("TIMEOUT", "yellow"),
               "aborted": ("ABORT", "yellow"), "broken": ("BROKEN", "red")}


def render(store: Store, st: Style, show_all: bool = False) -> list[str]:
    """The complete output of `tm ls`."""
    out: list[str] = []
    sessions = set(runner.list_sessions())
    # runs/ is read before archive/: a run tm archives in between is then dropped by
    # the first read and found by the second, never missed by both. Keyed by
    # directory, so one seen in both is listed once. A finished run still in runs/
    # (tm has not archived it yet) belongs with the history.
    current = store.current()
    active = [r for r in current if not r.done]
    leftover = [r for r in current if r.done]
    # Only what is shown is read from archive/, so tm ls does not slow down as the
    # history grows either; the count of what is hidden comes from the listing alone.
    shown = None if show_all else RECENT_SHOWN
    by_dir = {r.path.name: r for r in store.archived(shown) + leftover}
    finished = sorted(by_dir.values(), key=lambda r: r.path.name, reverse=True)[:shown]
    hidden = max(0, store.archived_count() + len(leftover) - len(finished))

    holder = store.lock_holder()
    line = st.bold("tm: ") + (st.green(holder) if holder else st.dim("not running"))
    if store.paused():
        line += st.yellow("   [queue paused · tm resume]")
    out.append(line)

    # ---- running ---------------------------------------------------------------
    # Each table is laid out as plain text first, with a dim header row, then styled
    if active:
        out.append("")
        out.append(st.bold("RUNNING"))
        rows, tails = [["list", "gpu", "step", "time"]], [""]
        for run in active:
            done, _, _ = run.scan()
            tasks = run.tasks
            idx = done + 1
            # A now: run says so, since nothing else shows it skipped the queue. "-" is
            # a run from before now: existed that tm booked no card for.
            gpu = describe_gpus(run.gpus, "cpu" if run.start == "now" else "-")
            if run.start == "now" and run.gpus:
                gpu = "now:" + gpu
            step = tasks[idx - 1].name if idx <= len(tasks) else "-"
            rows.append([run.name, gpu, f"[{idx}/{len(tasks)}] {step}",
                         _elapsed(run.started)])

            tail = ""
            if idx <= len(tasks) and tasks[idx - 1].session:
                sess = runner.Session(tasks[idx - 1].session)
                if sess.name in sessions:
                    silent = sess.silent_for()
                    if silent is not None and silent >= SILENT_WARN:
                        tail += st.yellow(f"  ⚠ silent {fmt_duration(silent)}")
                    tail += st.dim(f"  -> tmux attach -t {sess.name}")
                else:
                    tail += st.red("  session gone")
            tails.append(tail)
        for i, (cells, tail) in enumerate(zip(_columns(rows, right=frozenset({3})), tails)):
            if i == 0:
                out.append(st.dim("  " + "  ".join(cells).rstrip()))
                continue
            name, gpu, step, took = cells
            out.append(f"  {st.bold(name)}  {st.cyan(gpu)}  {step}  {took}{tail}")

    # ---- queued ----------------------------------------------------------------
    queued = store.queued()
    if queued:
        out.append("")
        out.append(st.bold("QUEUED") + st.dim("   (order = priority; rename to change it)"))
        rows, bad = [["file", "tasks", "start"]], [False]
        for path in queued:
            try:
                plan = load_plan(path)
                rows.append([path.name, f"{len(plan.tasks)} tasks",
                             describe_start(plan.start, short=True)])
                bad.append(False)
            except ConfigError as exc:
                rows.append([path.name, "", "BAD: " + str(exc).splitlines()[0]])
                bad.append(True)
        for i, cells in enumerate(_columns(rows)):
            if i == 0:
                out.append(st.dim("  " + "  ".join(cells).rstrip()))
                continue
            name, ntasks, note = cells
            note = st.red(note.rstrip()) if bad[i] else st.dim(note.rstrip())
            out.append(f"  {name}  {ntasks}  {note}")

    # ---- finished --------------------------------------------------------------
    # Newest first, capped at RECENT_SHOWN. The cut is always announced: a list that
    # stops silently hides the run you were looking for.
    if finished:
        out.append("")
        out.append(st.bold("RECENT"))
        rows, colours, tails = [["state", "list", "steps", "took", "finished"]], [""], [""]
        for run in finished:
            done, fail_i, fail_rc = run.scan()
            tasks = run.tasks
            mark, colour = _STATE_MARK.get(run.state, (run.state, ""))
            rows.append([mark, run.name, f"{done}/{len(tasks)}",
                         _elapsed(run.started, run.finished), run.finished[5:16]])
            colours.append(colour)
            tail = ""
            if fail_i and fail_i <= len(tasks):
                rec = tasks[fail_i - 1]
                tail += f"  {st.red(f'step{fail_i} {rec.name} rc={fail_rc}')}"
                if rec.session in sessions:
                    tail += st.dim(f"  -> tmux attach -t {rec.session}")
            tails.append(tail)
        laid = _columns(rows, right=frozenset({2, 3}))
        for i, (cells, colour, tail) in enumerate(zip(laid, colours, tails)):
            if i == 0:
                out.append(st.dim("  " + "  ".join(cells).rstrip()))
                continue
            mark, name, steps, took, ended = cells
            mark = getattr(st, colour)(mark) if colour else mark
            out.append(f"  {mark}  {name}  {steps}  {took}  {st.dim(ended)}{tail}")
    if hidden:
        out.append(st.dim(f"  ... {hidden} more, tm ls -a for all"))

    # ---- gpus ------------------------------------------------------------------
    try:
        gpus = query_gpus()
        out.append("")
        out.append(st.bold("GPUS"))
        for g in gpus:
            bar = (st.red if g.free_gib < LOW_FREE_GIB else
                   st.yellow if g.util > BUSY_UTIL_PCT else st.green)
            used = g.total_gib - g.free_gib
            out.append(f"  gpu{g.index}: {bar(f'{used:6.1f}')}/{g.total_gib:.1f} GiB used"
                       f"   util {g.util:3d}%")
    except ConfigError as exc:
        out.append("")
        out.append(st.yellow(f"GPUS: {exc}"))

    if not active and not queued:
        out.append("")
        out.append(st.dim(f"Queue is empty. Drop a yaml in: cp list.yaml {store.queue_dir}/"))
    return out
