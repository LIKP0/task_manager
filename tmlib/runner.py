"""The tmux layer: put a task in a tmux session, then watch it across a process boundary.

tm is not the task's parent process. The bash inside the tmux pane is what actually
wait()s for the exit code, and it writes that code to a file. That file is the only
channel between them. The wrapper is not spliced onto a tmux command line; it is
written out as `NN.sh` in the run directory, and tmux is handed only `bash NN.sh`
(see write_script).

That gives three states, checked once per tick:

    session alive + no rc  ->  still running
    rc present             ->  finished; the exit code is the file contents
                               (on failure the session stays up so you can attach)
    session gone  + no rc  ->  died without reporting (kill -9, OOM killer, reboot)

The third state is what holds the design up: no evidence means failure. Without it a
hard-killed train reads as success and test runs against half a checkpoint.

False success must be impossible, so every path leans the same way: cannot write to
disk -> no rc -> failure; hard-killed -> no rc -> failure. The rc file lands via
rename, so a half-written read cannot happen. The only way to fool it is a command
that swallows its own failure (a pipe, backgrounding), which is why the wrapper sets
`pipefail`.

The three-state decision is not made here. This module answers one question, "is the
session alive" (`Session.alive()`). Reading the rc file belongs to `store.Run.rc()`,
since the run directory owns it; two implementations would eventually disagree about
something like whether an empty file counts. The two halves meet in `tm._advance`:

    if not sess.alive() and run.rc(idx) is None:   -> LOST

Session first, then re-check rc. The wrapper writes rc before exiting, so if the
session is gone and there is still no rc, it really was never written.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

# Shell that runs the wrapper script. bash is required for `set -o pipefail`, which
# dash/sh lack; without it `train.py | tee log` reports rc=0 even when train crashes
# (measured). If bash is missing we fall back to sh and drop pipefail: degraded, but
# not a syntax error.
BASH = shutil.which("bash")
SCRIPT_SHELL = BASH or "/bin/sh"

# Resolved once, like BASH above. It is also the only thing separating sessions tm
# may kill from the user's own, so it lives next to the function that builds names.
TMUX = shutil.which("tmux")

# Every session tm creates starts with this. `tm clean` uses it to find orphans.
SESSION_PREFIX = "tm-"

# Interactive shell that pins the pane open after a failure. bash reads ~/.bashrc, so
# conda is live when you attach.
HOLD_SHELL = "bash" if BASH else "sh"


class TmuxError(RuntimeError):
    pass


def _tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if not TMUX:
        raise TmuxError("tmux not found — tm runs every task inside a tmux session")
    res = subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=30)
    if check and res.returncode != 0:
        raise TmuxError(f"tmux {' '.join(args)}: {res.stderr.strip() or res.returncode}")
    return res


def session_name(run_name: str, run_id: str, index: int, task_name: str) -> str:
    """tm-ccfm_c-20260808-143022-02-test — what to attach to, without a lookup.

    `run_id` distinguishes two runs of the same list. Without it, a failed run's
    pinned pane blocks re-queueing that list: launch() refuses to reuse a live
    session name, so the retry is aborted before it runs a single step.
    """
    return f"{SESSION_PREFIX}{run_name}-{run_id}-{index:02d}-{task_name}"


def list_sessions() -> list[str]:
    res = _tmux("list-sessions", "-F", "#{session_name}", check=False)
    if res.returncode != 0:          # no server = no sessions, not an error
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def tm_sessions() -> list[str]:
    """Only the sessions tm created. What `tm clean` is allowed to consider killing."""
    return [s for s in list_sessions() if s.startswith(SESSION_PREFIX)]


def write_script(path: Path, cmd: str, cwd: Path, rc_file: Path, label: str) -> None:
    """Write one step as a self-contained script — the thing tmux actually runs.

    Writing a file instead of splicing onto a tmux command line buys four things:
      1. Shell independence. tmux executes command strings with its default-shell,
         which cannot be assumed to be POSIX.
      2. Room for `set -o pipefail` and multi-line code, with no nested quoting.
      3. A record of what actually ran; reproducing a step is `bash NN.sh`.
      4. Like the rc file and the queue directory, state lives on the filesystem
         rather than only in memory.
    """
    rc_tmp = Path(f"{rc_file}.tmp")
    lines = [
        f"#!{SCRIPT_SHELL}",
        f"# tm: {label}",
        "# Everything tm hands to tmux. Run it directly to reproduce this step.",
    ]
    if BASH:
        # A pipeline's exit status is that of its last command, so in
        # `python train.py | tee log` a crashed train with a successful tee still
        # gives $? == 0. tm would read the failure as success and run test against
        # half a checkpoint. pipefail corrects it.
        lines.append("set -o pipefail")
    lines += [
        # The script cds itself, so running it standalone is also correct
        # (tmux -c only sets the pane's initial cwd).
        f"cd {shlex.quote(str(cwd))} || exit 1",
        "",
        # Run the command in a subshell. Without it, a command that calls `exit`
        # (or ends in `exec`) takes the wrapper with it, the rc file is never
        # written, and tm reads an ordinary failure as "no evidence = LOST".
        # The subshell contains the exit; $? is still the real value.
        f"( {cmd} )",
        "rc=$?",
        "",
        # Write a temp file, then rename. rename is atomic, so the rc file is either
        # absent or complete: a torn read goes from "handled" to "impossible".
        # If the write fails, && short-circuits and no rc file is left, so tm judges
        # LOST (a failure) — the safe direction.
        f"echo $rc > {shlex.quote(str(rc_tmp))} && "
        f"mv {shlex.quote(str(rc_tmp))} {shlex.quote(str(rc_file))}",
        "",
        # Succeed and vanish (leaving no junk sessions); fail and stay pinned with
        # full scrollback, as an interactive shell in the job's cwd and environment.
        "[ $rc -eq 0 ] && exit 0",
        f"exec {HOLD_SHELL} -i",
        "",
    ]
    path.write_text("\n".join(lines))
    path.chmod(0o755)


class Session:
    """One tmux session = one task, running or already dead.

    Only answers tmux questions: is it alive, what is on screen, how long since
    output. Whether the step finished belongs to the rc file, read by `store.Run.rc()`.
    """

    def __init__(self, name: str, rc_file: Path | None = None):
        self.name = name
        self.rc_file = rc_file          # only launch() needs it (goes into the script)

    # A leading '=' forces an exact match; tmux matches by prefix otherwise, so
    # tm-x-01-a would hit tm-x-01-abc. But '=' is only understood by session-level
    # commands. Pane-level commands like capture-pane and display-message reject it,
    # hence two spellings: exact for liveness checks, bare name for screen reads.
    #
    # Do not "simplify" the two pane_target uses below to target:
    # `display-message -t '=name'` kills the entire tmux server on 3.2a (reproduces
    # every time, whether or not the session exists), and silent_for() runs on every
    # `tm ls` — one `tm ls` would take down every running job.
    @property
    def target(self) -> str:
        return f"={self.name}"

    @property
    def pane_target(self) -> str:
        # A bare name is an exact hit when the session exists (tmux tries exact
        # before prefix), and these two call sites only affect display; they take no
        # part in deciding whether a step finished.
        return self.name

    def launch(self, cmd: str, cwd: Path, env: dict[str, str], script: Path) -> None:
        """Start the session. The command writes its exit code to rc_file when done.

        `script` is where the wrapper lands (`NN.sh` in the run directory).
        """
        if self.rc_file is None:
            raise TmuxError(f"session {self.name}: launch() requires rc_file")
        if self.alive():
            raise TmuxError(f"session {self.name} already exists — "
                            f"kill it first: tmux kill-session -t {self.name}")
        write_script(script, cmd, cwd, self.rc_file, self.name)
        args = ["new-session", "-d", "-s", self.name, "-c", str(cwd)]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        # This one line is all tmux gets. tmux runs it with default-shell (normally
        # $SHELL), and `bash <path>` is a valid command invocation in fish, csh and
        # zsh alike. Keeping the wrapper's shell syntax inside the file means we
        # never have to assume the user's login shell is POSIX.
        args.append(f"{SCRIPT_SHELL} {shlex.quote(str(script))}")
        _tmux(*args)

    def alive(self) -> bool:
        return _tmux("has-session", "-t", self.target, check=False).returncode == 0

    def silent_for(self) -> float | None:
        """Seconds since this pane last produced output, or None if unavailable.

        Used to flag stuck jobs in yellow. It only warns, never acts: checkpointing
        and CPU-bound eval are silent for long stretches, so an automatic kill would
        eventually hit a healthy job.

        `#{window_activity}` updates without enabling `monitor-activity` (measured on
        3.2a: on a fresh server a chatty session reads 0s and a quiet one grows).
        """
        res = _tmux("display-message", "-p", "-t", self.pane_target,
                    "#{window_activity}", check=False)
        if res.returncode != 0:
            return None
        try:
            return max(0.0, time.time() - int(res.stdout.strip()))
        except ValueError:
            return None

    def capture(self, lines: int = 40) -> list[str]:
        """Grab the current screen, to show the last lines of a failure in place."""
        res = _tmux("capture-pane", "-p", "-t", self.pane_target, check=False)
        if res.returncode != 0:
            return []
        out = [ln.rstrip() for ln in res.stdout.splitlines()]
        while out and not out[-1]:
            out.pop()
        return out[-lines:]

    def kill(self) -> None:
        _tmux("kill-session", "-t", self.target, check=False)
