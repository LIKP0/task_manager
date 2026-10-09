"""The on-disk state layer.

All of tm's truth lives here; nothing authoritative is held in memory. tm can be
killed and restarted at any time, and you can edit these files by hand while it is
not running.

    <repo>/                         where tm.py sits, not ~/.tm
      tm_config.yaml                tm's own settings (see settings.py)
      lock                          flock, so two terminals cannot both run tm
      paused                        present = do not scan the queue (tm hold/resume)
      queue/
        010_ccfm_c.yaml             here = not started. Copying one in == `tm add`
        020_ccfm_d.yaml
      runs/
        20260808_143022_ccfm_c/     here = in progress. Moved out of queue/ at start
          list.yaml                 the original, moved out of queue/
          run.yaml                  written by tm: state, GPUs held, session names
          01.rc  02.rc              exit codes written by the tasks themselves —
                                    the only evidence a step finished
          events.log                a human-readable log
      archive/
        20260807_090000_ccfm_b/     here = finished. Moved out of runs/ by tm once it
                                    records the final state; never touched again

The directory a list sits in says where it is in its life: queue/, runs/, archive/.
That keeps the scheduler's per-tick read to runs/, so it costs the same however long
the history is.

The key split: `run.yaml` is the plan, written by tm. `NN.rc` is the result, written
by the task. rc files keep appearing after tm dies, and tm does not get a say in what
they contain.

They live in the repo rather than ~/.tm because the queue and run records are things
you read often, and a dotdir in $HOME is awkward to browse. The cost is that they sit
in a git repo, so all of it is gitignored: runtime state does not belong in version
control.
"""

from __future__ import annotations

import errno
import fcntl
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .config import NowSpec, Plan, SafeLoader, resolve

# The on-disk timestamp format. Written here and by tm.py, read back by view.py —
# one constant so a change cannot half-land (view._elapsed swallows a parse failure
# and silently shows a blank duration).
TIME_FMT = "%Y-%m-%d %H:%M:%S"

# State lives in the repo directory (one level above tmlib/), not ~/.tm.
# Derived from __file__ rather than cwd: tm is symlinked into PATH and invoked from
# anywhere, and .resolve() follows the symlink, so this always points at the repo.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# Queue filenames carry a 3-digit sequence number, and queue order is filename order:
# '1000_x' sorts before '990_y', so anything past this would land out of order.
MAX_SEQ = 999

# Terminal run states. Once here, a run is never touched again.
TERMINAL = {"done", "failed", "lost", "timeout", "aborted", "broken"}


class StoreError(Exception):
    """The state directory cannot be used (unwritable, and so on).

    Named rather than a bare RuntimeError so the handler cannot also swallow
    TmuxError, which subclasses RuntimeError, or an unrelated bug.
    """


class LockBusy(Exception):
    """Another tm is already running."""

    def __init__(self, holder: str):
        super().__init__(holder)
        self.holder = holder


@dataclass
class TaskRecord:
    """One step in run.yaml. cmd is final: variables are already expanded."""
    name: str
    cmd: str
    session: str = ""
    started: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "cmd": self.cmd,
                "session": self.session, "started": self.started}


class Run:
    """One execution of a list. Every read and write hits disk; this is just a cursor."""

    def __init__(self, path: Path):
        self.path = path
        self._doc: dict = {}
        self.reload()

    # ---- reads --------------------------------------------------------------
    def reload(self) -> None:
        try:
            doc = yaml.load(self.run_yaml.read_text(), Loader=SafeLoader)
        except (OSError, yaml.YAMLError):
            doc = None
        self._doc = doc if isinstance(doc, dict) else {}

    @property
    def name(self) -> str:
        # run.yaml always has a name (written at claim time, and name is required in
        # the list). Falling back to the directory name only guards a deleted or
        # corrupted run.yaml; it is not the normal path.
        return str(self._doc.get("name") or self.path.name)

    @property
    def state(self) -> str:
        return str(self._doc.get("state") or "pending")

    @property
    def gpus(self) -> list[int]:
        return list(self._doc.get("gpus") or [])

    @property
    def cwd(self) -> str:
        return str(self._doc.get("cwd") or "")

    @property
    def gpu_budget_gb(self) -> float:
        """VRAM this run declared it needs, per card. Sharers bill against this."""
        try:
            return float(self._doc.get("gpu_budget_gb") or 0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def exclusive(self) -> bool:
        # A missing key means an old run (from before exclusive existed); treating it
        # as exclusive is what keeps it from being crowded out.
        return bool(self._doc.get("exclusive", True))

    @property
    def start(self) -> str:
        """'wait' or 'now', as the list asked; '' for a run from before now: existed."""
        return str(self._doc.get("start") or "")

    @property
    def started(self) -> str:
        return str(self._doc.get("started") or "")

    @property
    def finished(self) -> str:
        return str(self._doc.get("finished") or "")

    @property
    def done(self) -> bool:
        return self.state in TERMINAL

    @property
    def tasks(self) -> list[TaskRecord]:
        out = []
        for item in self._doc.get("tasks") or []:
            if isinstance(item, dict):
                out.append(TaskRecord(name=str(item.get("name", "?")),
                                      cmd=str(item.get("cmd", "")),
                                      session=str(item.get("session", "")),
                                      started=str(item.get("started", ""))))
        return out

    @property
    def run_yaml(self) -> Path:
        """tm's plan for this run. Written by save(), never by a task."""
        return self.path / "run.yaml"

    @property
    def run_id(self) -> str:
        """Short tmux-safe token telling this run apart from another of the same list.

        Run directories are <YYYYMMDD>_<HHMMSS>_<name>, plus `.N` when two start in
        the same second. The date is kept, not just the time: a failed step pins its
        pane open indefinitely, so a time-only id would collide again with a retry
        that happens to start at the same second on a later day — which is the exact
        abort this id exists to prevent.
        """
        name = self.path.name
        # List names cannot contain '.', so a trailing .N is claim()'s same-second
        # disambiguator and nothing else.
        _, dot, tail = name.rpartition(".")
        parts = name.split("_")
        stamp = f"{parts[0]}-{parts[1]}" if len(parts) > 2 else name
        return f"{stamp}-{tail}" if dot and tail.isdigit() else stamp

    def rc_path(self, index: int) -> Path:
        """Exit-code file for step `index` (1-based). The task writes it itself."""
        return self.path / f"{index:02d}.rc"

    def script_path(self, index: int) -> Path:
        """Wrapper script for step `index` — what tmux actually runs.

        Keeping it in the run directory is useful afterwards: it shows exactly what
        ran, and reproducing that step is `bash NN.sh`.
        """
        return self.path / f"{index:02d}.sh"

    def rc(self, index: int) -> int | None:
        """Read the exit code for step `index`; None if it has not been written yet."""
        try:
            text = self.rc_path(index).read_text().strip()
        except (OSError, ValueError):
            return None
        try:
            return int(text)
        except ValueError:
            # File present, contents not a number. The wrapper writes .tmp then
            # renames, so this is unreachable normally; it guards hand-edits and
            # oddities, still leaning the same way: unparseable means failure.
            return 1 if text else None

    def scan(self) -> tuple[int, int | None, int | None]:
        """(steps that succeeded, index of the failed step, its exit code).

        Counted entirely from rc files, not from run.yaml: the tasks write those
        themselves, so a tm restart does not disturb them. Failure stops the list, so
        no rc files exist past the failing step.
        """
        for i in range(1, len(self.tasks) + 1):
            rc = self.rc(i)
            if rc is None:
                return i - 1, None, None      # step i is running, or not started
            if rc != 0:
                return i - 1, i, rc
        return len(self.tasks), None, None

    def current_index(self) -> int:
        """Which step is running or due to start, 1-based. len+1 when all are done."""
        return self.scan()[0] + 1

    # ---- writes -------------------------------------------------------------
    def save(self, **fields) -> None:
        self._doc.update(fields)
        tmp = self.run_yaml.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True))
        tmp.replace(self.run_yaml)              # atomic swap; no torn reads

    def set_state(self, state: str, **fields) -> None:
        if state in TERMINAL:
            fields.setdefault("finished", now_stamp())
        self.save(state=state, **fields)
        self.event(f"state -> {state}")

    def set_task(self, index: int, **fields) -> None:
        tasks = list(self._doc.get("tasks") or [])
        if 1 <= index <= len(tasks) and isinstance(tasks[index - 1], dict):
            tasks[index - 1].update(fields)
            self.save(tasks=tasks)

    def event(self, text: str) -> None:
        """Append to the human-readable log. Never raises.

        events.log is a convenience, not evidence: nothing reads it back. A failure
        here used to propagate out of claim() and make a run that had *already* been
        created look like one that never started, so the caller skipped booking its
        cards and a later list could be put on the same GPU.
        """
        try:
            with (self.path / "events.log").open("a") as fh:
                fh.write(f"{now_stamp()}  {text}\n")
        except OSError:
            pass


class Store:
    """The state directory tree (the repo by default).

    All disk access goes through here, so tests only need a different root.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get("TM_ROOT") or DEFAULT_ROOT).expanduser()
        self.queue_dir = self.root / "queue"
        self.runs_dir = self.root / "runs"
        self.archive_dir = self.root / "archive"

    def ensure(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def check_writable(self) -> None:
        """Confirm we can write to disk before starting anything.

        If rc files cannot be written, even successful tasks are judged failures
        (no rc = died without reporting). Better to fail now than halfway through.
        """
        # ensure() has to be inside the try: if the root cannot be created, mkdir
        # raises PermissionError, and leaving it outside means the user gets a
        # traceback instead of the message below.
        # Probe every directory tm actually writes into, not just the root. ensure()
        # succeeds when the subdirectories already exist, so a writable root with a
        # read-only runs/ would start cleanly and then fail on every single claim.
        try:
            self.ensure()
        except OSError as exc:
            # Not necessarily a permission problem: a stray *file* named runs/ next
            # to tm.py raises FileExistsError here, and blaming the root for that
            # sends you looking in the wrong place.
            raise StoreError(f"cannot create the state directories under "
                             f"{self.root}: {exc}") from exc
        for d in (self.root, self.queue_dir, self.runs_dir, self.archive_dir):
            probe = d / ".writable"
            try:
                probe.write_text("ok")
                probe.unlink()
            except OSError as exc:
                raise StoreError(f"{d} is not writable: {exc}") from exc

    # ---- single-instance lock -----------------------------------------------
    @property
    def lock_path(self) -> Path:
        return self.root / "lock"

    @contextmanager
    def lock(self):
        """Process-level mutex. The kernel releases it on death, so it never sticks."""
        self.ensure()
        fh = self.lock_path.open("a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            # Not ours, so the file is left exactly as the holder wrote it: clearing
            # it here made `tm ls` report a running tm as not running.
            try:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                fh.seek(0)
                raise LockBusy(fh.read().strip() or "unknown") from None
            finally:
                fh.close()
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid {os.getpid()} since {now_stamp()}\n")
            fh.flush()
            yield
        finally:
            try:
                fh.seek(0)
                fh.truncate()
                fh.flush()
            except OSError:
                pass
            fh.close()          # closing releases the flock

    def lock_holder(self) -> str:
        """Who holds the lock right now; empty means nobody is running.

        The test is whether the lock can be taken non-blocking, not what the file
        says. When tm is hard-killed the kernel reclaims the flock but the text
        stays behind, so reading alone would point at a long-dead pid and call it
        running.
        """
        path = self.lock_path
        try:
            text = path.read_text().strip()
        except OSError:
            return ""
        if not text:
            return ""
        try:
            with path.open("a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return text            # cannot take it = a tm really holds it
                fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            return text                    # cannot even open it; trust the text
        return ""                          # took it = the last tm was hard-killed

    # ---- pause ---------------------------------------------------------------
    # While this file exists tm starts nothing new from the queue (running tasks are
    # unaffected). It gives you a quiet window to reorder or edit the queue without
    # something being picked up mid-edit.
    @property
    def _pause_flag(self) -> Path:
        return self.root / "paused"

    def paused(self) -> bool:
        return self._pause_flag.exists()

    def pause(self) -> None:
        self.ensure()
        self._pause_flag.write_text(f"held at {now_stamp()}\n")

    def resume(self) -> bool:
        """Clear the flag. False if it was not set, so the caller can say so."""
        if not self.paused():
            return False
        self._pause_flag.unlink(missing_ok=True)
        return True

    # ---- queue ---------------------------------------------------------------
    def queued(self) -> list[Path]:
        """Pending lists, sorted by filename. That order is the priority."""
        if not self.queue_dir.is_dir():
            return []
        return sorted(p for p in self.queue_dir.iterdir()
                      if p.is_file() and p.suffix in (".yaml", ".yml"))

    def add(self, src: Path, seq: int | None = None) -> Path:
        """Copy a yaml into the queue. This is all `tm add` does; cp is equivalent."""
        self.ensure()
        if seq is None:
            used = [int(m) for p in self.queued()
                    if (m := p.stem.split("_")[0]).isdigit()]
            seq = (max(used) + 10) // 10 * 10 if used else 10
        # Queue filenames only carry ordering; the list's name is in the yaml
        dst = self.queue_dir / f"{seq:03d}_{src.stem}{src.suffix}"
        n = seq
        while dst.exists():                 # shift along on collision, never overwrite
            n += 1
            dst = self.queue_dir / f"{n:03d}_{src.stem}{src.suffix}"
        if not 0 <= n <= MAX_SEQ:
            raise StoreError(f"sequence number {n} is outside 0-{MAX_SEQ}, so it would "
                             f"not sort into place; renumber {self.queue_dir} first")
        # Write .tmp then rename. A plain copy has an intermediate state, and a tick
        # could read a half-written yaml. If the truncation lands on a task boundary
        # the result is still valid yaml, just missing steps — tm would silently run
        # a truncated list. rename is atomic, so the queue holds nothing or a whole
        # file. queued() filters by suffix, so the .tmp is invisible meanwhile.
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    # ---- run directories ------------------------------------------------------
    def claim(self, queue_path: Path, plan: Plan, gpus: list[int],
              state: str = "running", **fields) -> Run:
        """Turn a queued yaml into a run: make the directory, move the file, write run.yaml.

        Move rather than copy, so the queue only ever contains things that have not
        started. You can edit it by hand at any time without touching running work.

        A list skipped at claim time (timed out, failed the device check) passes its
        terminal `state` here, so run.yaml is written once, already final. Claiming
        it as running and then calling set_state() left a window where a failed
        second write meant a running run with no cards — which a restarted tm would
        launch with no CUDA_VISIBLE_DEVICES, free to use every card. Such a run never
        starts, so its directory is made in archive/ rather than passing through runs/.
        """
        self.ensure()
        # Expand commands before anything is created or moved. Everything below this
        # line changes the filesystem, so a failure here must not leave the list half
        # out of the queue with an empty run directory behind it.
        deferred = {"GPU": ",".join(str(g) for g in gpus)}
        records = [TaskRecord(name=t.name, cmd=resolve(t.cmd, deferred)).to_dict()
                   for t in plan.tasks]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent = self.archive_dir if state in TERMINAL else self.runs_dir
        path = parent / f"{stamp}_{plan.name}"
        n = 1
        # Two starts within the same second. Both directories count: the same name in
        # runs/ and archive/ would stop the one in runs/ from being archived.
        while (self.runs_dir / path.name).exists() or (self.archive_dir / path.name).exists():
            n += 1
            path = parent / f"{stamp}_{plan.name}.{n}"
        path.mkdir(parents=True)

        shutil.move(str(queue_path), str(path / "list.yaml"))

        # The GPUs are fixed at this moment, so commands are expanded once here and
        # run.yaml stores the exact line handed to tmux. After a restart tm just
        # follows it; list.yaml is never re-read.
        run = Run(path)
        run.save(
            name=plan.name,
            state=state,
            cwd=str(plan.cwd),
            gpus=list(gpus),
            gpu_budget_gb=plan.start.gpu_free_gb,
            exclusive=plan.start.exclusive,
            start="now" if isinstance(plan.start, NowSpec) else "wait",
            started=now_stamp(),
            finished=now_stamp() if state in TERMINAL else "",
            tasks=records,
            **fields,
        )
        run.event(f"claimed from {queue_path.name}, gpus={gpus or 'none'}")
        if state != "running":
            run.event(f"state -> {state}")
        return run

    def _load(self, d: Path, limit: int | None = None) -> list[Run]:
        """The runs under d, newest first. Only the first `limit` are read."""
        if not d.is_dir():
            return []
        paths = sorted((p for p in d.iterdir() if p.is_dir()), reverse=True)[:limit]
        # A directory can leave between the listing and the read: tm archives finished
        # runs, and `tm prune` deletes archived ones. Its run.yaml then reads as
        # missing — a running list with no tasks. Checked after the read, so a move
        # cannot slip between.
        return [r for r in (Run(p) for p in paths) if r.path.is_dir()]

    def current(self) -> list[Run]:
        """Everything in runs/: the runs in progress, plus any finished run that has
        not been archived yet (a tm died in between). Normally a handful."""
        return self._load(self.runs_dir)

    def active(self) -> list[Run]:
        """Runs not yet in a terminal state."""
        return [r for r in self.current() if not r.done]

    def archived(self, limit: int | None = None) -> list[Run]:
        """Finished runs, newest first. Only the first `limit` are read."""
        return self._load(self.archive_dir, limit)

    def archived_count(self) -> int:
        if not self.archive_dir.is_dir():
            return 0
        return sum(1 for p in self.archive_dir.iterdir() if p.is_dir())

    def archive(self, run: Run) -> None:
        """Move a finished run into archive/, out of every later tick's read.

        One rename, and the directory keeps its name, so the run id and the session
        names derived from it stay the same.
        """
        target = self.archive_dir / run.path.name
        run.path.rename(target)
        run.path = target

    def remove(self, run: Run) -> None:
        """Delete an archived run's directory. Anything else is refused.

        Renamed out of archive/ first, so `tm ls` never reads it half-deleted.
        Deleted in place, it would lose run.yaml while still listed and show up as a
        run that never started.
        """
        if run.path.parent != self.archive_dir:
            raise StoreError(f"{run.path.name} is not in {self.archive_dir}")
        trash = self.root / ".pruning"
        trash.mkdir(exist_ok=True)
        # Anything already here is out of archive/ and left by a delete that failed
        # part-way. Nothing else will ever finish it, so do that before touching this
        # run; if it fails again, the error names the leftover and this run is intact.
        for left in list(trash.iterdir()):
            shutil.rmtree(left)
        target = trash / run.path.name
        run.path.rename(target)
        shutil.rmtree(target)
        trash.rmdir()


def now_stamp() -> str:
    """The one place a timestamp is formatted for disk."""
    return datetime.now().strftime(TIME_FMT)
