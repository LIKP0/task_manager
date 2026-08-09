"""task list 的解析：yaml -> Plan。

这一层不碰 tmux、不碰显卡、不碰磁盘状态，只负责把一份 yaml 变成
一个「可以执行的计划」，或者告诉你它哪里写错了。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MIB_PER_GIB = 1024

# 命令里的 `--config <path>`，用于开跑前检查 trainer.devices
_CONFIG_ARG_RE = re.compile(r"--config[=\s]+(\S+)")
_CUDA_ARG_RE = re.compile(r"--cuda[=\s]+(\S+)")

# 这些占位符在读 yaml 时不展开，要等选好卡之后才知道值
DEFERRED_VARS = {"GPU"}

# 队列文件名的排序前缀：010_ccfm_c.yaml -> ccfm_c
_SEQ_PREFIX_RE = re.compile(r"^(\d+)[_-]")

# tmux 会话名不能带 ':' 和 '.'（它们是 target 语法的分隔符），顺手也挡掉空格
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


class ConfigError(Exception):
    """task list 写得不对，或者环境不满足。"""


@dataclass
class Task:
    name: str
    cmd: str


@dataclass
class WaitSpec:
    """上机条件。没有 gpu_free_gb 就表示不管显卡，直接开跑。"""

    gpu_free_gb: float | None = None
    gpus: int = 1
    gpu_index: list[int] | None = None   # None 表示 any
    stable_for: float = 120.0
    timeout: float | None = None         # 等这么久还没卡就放弃这个 list
    exclusive: bool = True               # 这张卡不许再放第二个 tm run

    @property
    def manages_gpu(self) -> bool:
        return self.gpu_free_gb is not None


@dataclass
class Plan:
    tasks: list[Task]
    cwd: Path
    name: str
    wait: WaitSpec = field(default_factory=WaitSpec)
    source: Path | None = None           # 从哪个 yaml 读出来的


# --------------------------------------------------------------------------- #
# 变量替换
# --------------------------------------------------------------------------- #

def substitute(cmd: str, variables: dict[str, str], where: str) -> str:
    """把 `{KEY}` 替换成 vars 里的值。

    未定义的变量直接报错，不静默留原样——config 路径打错一个字母就白跑几小时。
    命令本身要用花括号时写 `{{` / `}}`，比如 python 的 f-string、awk 的 `{{print $1}}`。
    DEFERRED_VARS 里的名字原样留着，等选好卡之后再展开。
    """
    missing: list[str] = []

    def repl(m: re.Match[str]) -> str:
        if m.group(0) in ("{{", "}}"):
            return m.group(0)
        key = m.group(1)
        if key in DEFERRED_VARS:
            return m.group(0)
        if key not in variables:
            missing.append(key)
            return m.group(0)
        return variables[key]

    out = re.sub(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, cmd)
    if missing:
        raise ConfigError(
            f"{where}: undefined variable(s) {sorted(set(missing))}. "
            f"Define them under 'vars:' or pass -v KEY=VALUE "
            f"(write {{{{ }}}} for literal braces)."
        )
    return out


def unescape_braces(cmd: str) -> str:
    """展开最后剩下的 `{{` / `}}`，在命令真正执行前调用。"""
    return re.sub(r"\{\{|\}\}", lambda m: m.group(0)[0], cmd)


def resolve(cmd: str, gpu_value: str) -> str:
    """把 {GPU} 换成真实卡号，再展开转义的花括号。交给 tmux 之前的最后一步。"""
    return unescape_braces(cmd.replace("{GPU}", gpu_value))


# --------------------------------------------------------------------------- #
# wait: 块
# --------------------------------------------------------------------------- #

def _parse_wait(raw: object, where: str) -> WaitSpec:
    if raw is None:
        return WaitSpec()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: 'wait:' must be a mapping")

    known = {"gpu_free_gb", "gpus", "gpu_index", "stable_for", "timeout", "exclusive"}
    # YAML 1.1 把裸 on/off/yes/no 当布尔值，写 `on:` 会变成 True 这个键
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

    if raw.get("gpu_free_gb") is not None:
        try:
            spec.gpu_free_gb = float(raw["gpu_free_gb"])
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: 'gpu_free_gb' must be a number (GiB)") from None
        if spec.gpu_free_gb <= 0:
            raise ConfigError(f"{where}: 'gpu_free_gb' must be positive")

    for key, cast, check in (("gpus", int, lambda v: v >= 1),
                             ("stable_for", float, lambda v: v >= 0)):
        if raw.get(key) is not None:
            try:
                value = cast(raw[key])
            except (TypeError, ValueError):
                raise ConfigError(f"{where}: '{key}' must be a number") from None
            if not check(value):
                raise ConfigError(f"{where}: '{key}' out of range: {raw[key]!r}")
            setattr(spec, key, value)

    if raw.get("exclusive") is not None:
        if not isinstance(raw["exclusive"], bool):
            raise ConfigError(f"{where}: 'exclusive' must be true or false")
        spec.exclusive = raw["exclusive"]

    if raw.get("timeout") is not None:
        try:
            spec.timeout = float(raw["timeout"])
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: 'timeout' must be a number of seconds") from None

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
# 主入口
# --------------------------------------------------------------------------- #

def list_name(path: Path) -> str:
    """从文件名推 list 名字，剥掉队列用的排序前缀：010_ccfm_c.yaml -> ccfm_c"""
    return _SEQ_PREFIX_RE.sub("", path.stem)


def load_plan(path: Path, cli_vars: dict[str, str] | None = None,
              cli_cwd: Path | None = None) -> Plan:
    """读一份 task list，展开变量，返回可执行的计划。"""
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc

    if doc is None:
        raise ConfigError(f"{path}: file is empty")
    if isinstance(doc, list):          # 整份文件直接是个列表也认
        doc = {"tasks": doc}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping with 'tasks:', got {type(doc).__name__}")

    raw_tasks = doc.get("tasks")
    if raw_tasks is None:
        raise ConfigError(f"{path}: missing 'tasks:'")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ConfigError(f"{path}: 'tasks:' must be a non-empty list")

    variables = {str(k): str(v) for k, v in (doc.get("vars") or {}).items()}
    variables.update(cli_vars or {})    # -v 覆盖文件里的 vars

    tasks: list[Task] = []
    for i, item in enumerate(raw_tasks, start=1):
        where = f"{path}: tasks[{i}]"
        if isinstance(item, str):       # 简写：直接写一条命令
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

    if cli_cwd is not None:
        cwd = cli_cwd
    elif doc.get("cwd"):
        cwd = Path(substitute(str(doc["cwd"]), variables, f"{path}: cwd"))
    else:
        # 必填，没有缺省值。曾经默认取「yaml 所在目录」，但 `tm add` 会把 list
        # 拷进 queue/，于是同一份 yaml 在 `tm check` 时算出一个 cwd、真正排队执行时
        # 算出另一个（queue/ 自己）。命令里的相对路径会跟着一起漂，而且漂得很安静。
        raise ConfigError(
            f"{path}: missing 'cwd:' — 它决定 tmux pane 的起始目录，也就是命令里所有"
            f"相对路径（脚本、config、输出目录）的解析基准。\n"
            f"      写成绝对路径，例如：  cwd: ~/my_project")
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ConfigError(f"{path}: cwd not found: {cwd}")

    name = str(doc.get("name") or list_name(path))
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ConfigError(f"{path}: list name {name!r} — use letters, digits, "
                          f"'_', '-' only (it becomes part of a tmux session name)")

    return Plan(tasks=tasks, cwd=cwd, name=name,
                wait=_parse_wait(doc.get("wait"), f"{path}: wait"), source=path)


# --------------------------------------------------------------------------- #
# 开跑前的静态检查：config 里的 trainer.devices
# --------------------------------------------------------------------------- #

def check_device_settings(tasks: list[Task], cwd: Path, gpus: int) -> list[str]:
    """tm 接管显卡时，任务 config 里的 devices 必须是相对编号。

    tm 会设 CUDA_VISIBLE_DEVICES=<物理卡号>，子进程眼里就只有 gpus 张卡、
    编号 0..gpus-1。config 再写 `devices: [1]` 会直接报错找不到卡——
    而且是在等了几个小时之后才报，所以开跑前就查掉。
    返回问题列表，空列表表示没问题。
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
        cfg = yaml.safe_load(cfg_path.read_text())
    except (OSError, yaml.YAMLError):
        return []                      # 读不了就不管，跑起来自然会报
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
