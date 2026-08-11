"""Parsing task lists: yaml -> Plan.

This layer touches no tmux, no GPUs and no on-disk state. It turns a yaml file into
an executable plan, or tells you what is wrong with it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:                                    # libyaml is ~11x faster than the python loader
    from yaml import CSafeLoader as SafeLoader
except ImportError:                     # pyyaml built without libyaml
    from yaml import SafeLoader

# `--config <path>` inside a command, used for the pre-flight trainer.devices check
_CONFIG_ARG_RE = re.compile(r"--config[=\s]+(\S+)")
_CUDA_ARG_RE = re.compile(r"--cuda[=\s]+(\S+)")

# Placeholders left unexpanded at parse time; their value is only known once a GPU
# has been picked.
DEFERRED_VARS = frozenset({"GPU"})

# tmux session names cannot contain ':' or '.' (target syntax separators). This also
# rules out spaces.
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


class ConfigError(Exception):
    """The task list is wrong, or the environment does not satisfy it."""


@dataclass
class Task:
    name: str
    cmd: str


@dataclass
class WaitSpec:
    """Start conditions. No gpu_free_gb means GPUs are ignored and the list starts at once."""

    gpu_free_gb: float | None = None
    gpus: int = 1
    gpu_index: list[int] | None = None   # None means any
    stable_for: float = 120.0            # seconds
    timeout: float | None = None         # seconds; give up on this list after waiting
    exclusive: bool = True               # no second tm run may share the card

    @property
    def manages_gpu(self) -> bool:
        return self.gpu_free_gb is not None


@dataclass
class Plan:
    tasks: list[Task]
    cwd: Path
    name: str
    wait: WaitSpec = field(default_factory=WaitSpec)


# --------------------------------------------------------------------------- #
# Variable substitution
# --------------------------------------------------------------------------- #

def substitute(cmd: str, variables: dict[str, str], where: str,
               defer: frozenset[str] | tuple[()] = DEFERRED_VARS) -> str:
    """Replace `{KEY}` with its value from vars.

    An undefined variable is an error rather than a silent pass-through: one typo in a
    config path otherwise costs hours. Write `{{` / `}}` for literal braces, as in
    python f-strings or awk's `{{print $1}}`. Names in `defer` are left alone for a
    later pass; `resolve()` calls with `defer=()` to finish the job.
    """
    missing: list[str] = []

    def repl(m: re.Match[str]) -> str:
        if m.group(0) in ("{{", "}}"):
            return m.group(0)
        key = m.group(1)
        if key in defer:
            return m.group(0)
        if key not in variables:
            missing.append(key)
            return m.group(0)
        return variables[key]

    out = re.sub(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, cmd)
    if missing:
        raise ConfigError(
            f"{where}: undefined variable(s) {sorted(set(missing))}. "
            f"Define them under 'vars:' in this file "
            f"(write {{{{ }}}} for literal braces)."
        )
    return out


def unescape_braces(cmd: str) -> str:
    """Unescape the remaining `{{` / `}}`. Called just before a command runs."""
    return re.sub(r"\{\{|\}\}", lambda m: m.group(0)[0], cmd)


def resolve(cmd: str, deferred: dict[str, str]) -> str:
    """Expand the deferred placeholders, then unescape braces. Last step before tmux.

    Goes through `substitute()` rather than str.replace: `{GPU}` is a substring of the
    escaped `{{GPU}}`, so a raw replace would turn a documented literal brace into an
    expansion. The regex knows the difference.
    """
    return unescape_braces(substitute(cmd, deferred, "resolve", defer=()))


# --------------------------------------------------------------------------- #
# The wait: block
# --------------------------------------------------------------------------- #

def _parse_wait(raw: object, where: str) -> WaitSpec:
    if raw is None:
        return WaitSpec()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: 'wait:' must be a mapping")

    known = {"gpu_free_gb", "gpus", "gpu_index", "stable_for", "timeout", "exclusive"}
    # YAML 1.1 reads bare on/off/yes/no as booleans, so `on:` becomes the key True
    if True in raw or False in raw:
        raise ConfigError(f"{where}: YAML parses a bare 'on:' / 'off:' key as a boolean. "
                          f"Use 'gpu_index:' to pick which cards are eligible.")
    unknown = set(raw) - known
    if unknown:
        hint = ""
        if unknown & {"after_pid", "on_oom", "max_retries", "poll"}:
            hint = ("  ('after_pid' / 'on_oom' / 'max_retries' / 'poll' were removed — "
                    "put the earlier job in the queue instead, and a failed task now "
                    "just stops its own list)")
        raise ConfigError(f"{where}: unknown key(s) under 'wait:': {sorted(unknown)}. "
                          f"Known keys: {sorted(known)}{hint}")

    spec = WaitSpec()

    # One table for every numeric key, so a new one cannot pick up a different style
    # of validation by accident.
    for key, cast, ok, unit in (("gpu_free_gb", float, lambda v: v > 0,  " (GiB)"),
                                ("gpus",        int,   lambda v: v >= 1, ""),
                                ("stable_for",  float, lambda v: v >= 0, " (seconds)"),
                                ("timeout",     float, lambda v: v >= 0, " (seconds)")):
        if raw.get(key) is not None:
            try:
                value = cast(raw[key])
            except (TypeError, ValueError):
                raise ConfigError(f"{where}: '{key}' must be a number{unit}") from None
            if not ok(value):
                raise ConfigError(f"{where}: '{key}' out of range: {raw[key]!r}")
            setattr(spec, key, value)

    if raw.get("exclusive") is not None:
        if not isinstance(raw["exclusive"], bool):
            raise ConfigError(f"{where}: 'exclusive' must be true or false")
        spec.exclusive = raw["exclusive"]

    which = raw.get("gpu_index", "any")
    if which in (None, "any"):
        spec.gpu_index = None
    elif isinstance(which, int) and not isinstance(which, bool):
        spec.gpu_index = [which]
    elif isinstance(which, list) and all(isinstance(x, int) and not isinstance(x, bool)
                                         for x in which):
        spec.gpu_index = list(which)
    else:
        raise ConfigError(f"{where}: 'gpu_index' must be 'any', an index, "
                          f"or a list of indices")

    if spec.manages_gpu and spec.gpu_index is not None and len(spec.gpu_index) < spec.gpus:
        raise ConfigError(f"{where}: 'gpu_index' lists {len(spec.gpu_index)} gpu(s) "
                          f"but 'gpus' is {spec.gpus}")

    return spec


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def read_yaml(path: Path) -> object:
    """Parse a yaml file, turning read and parse failures into ConfigError.

    FileNotFoundError is deliberately left to propagate: a missing task list and a
    missing tm_config.yaml need very different advice, so each caller words its own.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    try:
        return yaml.load(text, Loader=SafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML\n{exc}") from exc


def load_plan(path: Path) -> Plan:
    """Read a task list, expand variables, return an executable plan."""
    try:
        doc = read_yaml(path)
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc

    if doc is None:
        raise ConfigError(f"{path}: file is empty")
    if isinstance(doc, list):          # a bare list for the whole file is accepted too
        doc = {"tasks": doc}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping with 'tasks:', got {type(doc).__name__}")

    raw_tasks = doc.get("tasks")
    if raw_tasks is None:
        raise ConfigError(f"{path}: missing 'tasks:'")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ConfigError(f"{path}: 'tasks:' must be a non-empty list")

    # The yaml file is the only source of vars. There is no command-line override:
    # tm run is a long-lived process and lists are queued at arbitrary times, so
    # "the -v tm started with" and "this list" never line up in time.
    variables = {str(k): str(v) for k, v in (doc.get("vars") or {}).items()}

    tasks: list[Task] = []
    for i, item in enumerate(raw_tasks, start=1):
        where = f"{path}: tasks[{i}]"
        if isinstance(item, str):       # shorthand: just a command string
            name, cmd = f"step{i}", item
        elif isinstance(item, dict):
            if "cmd" not in item:
                raise ConfigError(f"{where}: missing 'cmd'")
            cmd = item["cmd"]
            if not isinstance(cmd, str):
                raise ConfigError(f"{where}: 'cmd' must be a string")
            name = str(item.get("name") or f"step{i}")
        else:
            raise ConfigError(f"{where}: expected a string or a mapping with 'cmd'")

        cmd = cmd.strip()
        if not cmd:
            raise ConfigError(f"{where}: 'cmd' is empty")
        if not _SAFE_NAME_RE.fullmatch(name):
            raise ConfigError(f"{where}: task name {name!r} — use letters, digits, "
                              f"'_', '-' only (it becomes part of a tmux session name)")
        tasks.append(Task(name=name, cmd=substitute(cmd, variables, where)))

    if doc.get("cwd"):
        cwd = Path(substitute(str(doc["cwd"]), variables, f"{path}: cwd"))
    else:
        # Required, no default. It used to default to the yaml's own directory, but
        # `tm add` copies the list into queue/, so the same file resolved one cwd
        # under `tm check` and a different one (queue/ itself) when it actually ran.
        # Relative paths in the commands drifted with it, silently.
        raise ConfigError(
            f"{path}: missing 'cwd:' — it is the tmux pane's starting directory, and so\n"
            f"      the base for every relative path in your commands (scripts, configs,\n"
            f"      output directories). Write an absolute path, e.g.  cwd: ~/MyProject")
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ConfigError(f"{path}: cwd not found: {cwd}")

    # Required, never derived from the filename. Deriving it also meant stripping the
    # `010_` prefix that `tm add` adds — one implicit transform existing only to undo
    # another, whose result ends up in tmux session names. Now it is what you wrote.
    if not doc.get("name"):
        raise ConfigError(
            f"{path}: missing 'name:' — it appears in tm ls, in the run directory name\n"
            f"      and in tmux session names (tm-<name>-01-<task>). Letters, digits,\n"
            f"      '_' and '-' only, e.g.  name: myrun")
    name = str(doc["name"])
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ConfigError(f"{path}: list name {name!r} — use letters, digits, "
                          f"'_', '-' only (it becomes part of a tmux session name)")

    return Plan(tasks=tasks, cwd=cwd, name=name,
                wait=_parse_wait(doc.get("wait"), f"{path}: wait"))


# --------------------------------------------------------------------------- #
# Pre-flight static check: trainer.devices in the task's config
# --------------------------------------------------------------------------- #

def check_device_settings(tasks: list[Task], cwd: Path, gpus: int) -> list[str]:
    """When tm manages the GPUs, devices in the task config must be relative indices.

    tm sets CUDA_VISIBLE_DEVICES=<physical index>, so the child process sees only
    `gpus` cards numbered 0..gpus-1. A config saying `devices: [1]` then fails to find
    the card — hours into the queue wait. Catching it up front is cheap.
    Returns a list of problems; empty means fine.
    """
    want_list = list(range(gpus))
    problems: list[str] = []
    seen: set[Path] = set()

    for task in tasks:
        m = _CONFIG_ARG_RE.search(task.cmd)
        if m:
            raw = m.group(1).strip("'\"")
            cfg_path = (cwd / raw).resolve() if not os.path.isabs(raw) else Path(raw)
            if cfg_path not in seen and cfg_path.is_file():
                seen.add(cfg_path)
                problems += _check_one_config(cfg_path, task.name, want_list, gpus)

        for cm in _CUDA_ARG_RE.finditer(task.cmd):
            value = cm.group(1).strip("'\"")
            if value.isdigit() and int(value) >= gpus:
                problems.append(
                    f"task '{task.name}': --cuda {value} — tm remaps the card, "
                    f"so it must be 0 (physical index goes to CUDA_VISIBLE_DEVICES)")

    return problems


def _check_one_config(cfg_path: Path, task_name: str, want_list: list[int],
                      gpus: int) -> list[str]:
    try:
        cfg = yaml.load(cfg_path.read_text(), Loader=SafeLoader)
    except (OSError, yaml.YAMLError):
        return []                      # unreadable: let the job itself complain
    if not isinstance(cfg, dict):
        return []
    trainer = cfg.get("trainer")
    if not isinstance(trainer, dict) or "devices" not in trainer:
        return []

    devices = trainer["devices"]
    ok = devices == want_list or devices == gpus or (gpus == 1 and devices in (0, [0], "0"))
    if ok:
        return []
    want = f"[{', '.join(str(i) for i in want_list)}]"
    return [f"task '{task_name}': {cfg_path}\n"
            f"      trainer.devices is {devices!r}, must be {want} (or {gpus}) "
            f"— tm assigns the physical card via CUDA_VISIBLE_DEVICES"]
