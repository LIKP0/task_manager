#!/usr/bin/env python3
"""tm —— 按 task list 顺序执行命令，可以先等卡再上机。

一条命令跑完且退出码为 0，才跑下一条；任何一步失败就立即终止，
并把失败那步的命令、退出码、输出末尾原样报出来。

`wait:` 块让它变成一个抢卡脚本：先等某个进程结束、再等到显存够用，
然后设好 CUDA_VISIBLE_DEVICES 自动上机。

设计上刻意不做的事：不排队、不做后台常驻（用 screen）、不重试失败的任务、
不做多实例互斥（同时只挂一个 tm）。

用法见 README.md，或 `tm --help`。
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_TASKFILE = "task_list.yaml"
DEFAULT_LOG_DIR = Path.home() / ".tm" / "logs"
TAIL_LINES = 30          # 失败时回放的输出行数
MIB_PER_GIB = 1024

# 默认的 /bin/sh 在多数发行版上是 dash，`source` / `conda activate` 会挂
SHELL = os.environ.get("TM_SHELL") or ("/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh")

# 进度条 / 颜色控制序列，只在写日志时剥掉
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[=>()][0-9A-Za-z]?")

# 抢卡失败的特征。只认明确的显存不足，别的错误重抢没有意义。
_OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|"
    r"cuDNN error: CUDNN_STATUS_(?:NOT_INITIALIZED|ALLOC_FAILED)",
    re.IGNORECASE,
)

# 命令里的 `--config <path>`，用于开跑前检查 trainer.devices
_CONFIG_ARG_RE = re.compile(r"--config[=\s]+(\S+)")
_CUDA_ARG_RE = re.compile(r"--cuda[=\s]+(\S+)")

# 这些占位符在读 yaml 时不展开，要等选好卡之后才知道值
DEFERRED_VARS = {"GPU"}


class ConfigError(Exception):
    """task list 写得不对，或者环境不满足。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class Task:
    name: str
    cmd: str
    status: str = "pending"   # pending / ok / failed / interrupted / skipped
    returncode: int | None = None
    seconds: float = 0.0
    resolved: str = ""        # 真正交给 shell 的那一行，报错时要能直接复制重跑

    def reset(self) -> None:
        """重抢一轮之前把状态清干净。"""
        self.status, self.returncode, self.seconds = "pending", None, 0.0
        self.resolved = ""


@dataclass
class WaitSpec:
    after_pid: int | None = None
    gpu_free_gb: float | None = None
    gpus: int = 1
    gpu_index: list[int] | None = None   # None 表示 any
    stable_for: float = 120.0
    poll: float = 30.0
    timeout: float | None = None
    on_oom: str = "requeue"          # requeue | stop
    max_retries: int = 5             # 0 = 不限

    @property
    def manages_gpu(self) -> bool:
        return self.gpu_free_gb is not None

    @property
    def active(self) -> bool:
        return self.after_pid is not None or self.manages_gpu


@dataclass
class Plan:
    tasks: list[Task]
    cwd: Path
    name: str
    wait: WaitSpec = field(default_factory=WaitSpec)


@dataclass
class Gpu:
    index: int
    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def free_gib(self) -> float:
        return self.free_mib / MIB_PER_GIB


# --------------------------------------------------------------------------- #
# 解析 task list
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


def _parse_wait(raw: object, where: str) -> WaitSpec:
    if raw is None:
        return WaitSpec()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: 'wait:' must be a mapping")

    known = {"after_pid", "gpu_free_gb", "gpus", "gpu_index", "stable_for",
             "poll", "timeout", "on_oom", "max_retries"}
    # YAML 1.1 把裸 on/off/yes/no 当布尔值，写 `on:` 会变成 True 这个键
    if True in raw or False in raw:
        raise ConfigError(f"{where}: YAML parses a bare 'on:' / 'off:' key as a boolean. "
                          f"Use 'gpu_index:' to pick which cards are eligible.")
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) under 'wait:': {sorted(unknown)}. "
                          f"Known keys: {sorted(known)}")

    spec = WaitSpec()

    if raw.get("after_pid") is not None:
        try:
            spec.after_pid = int(raw["after_pid"])
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: 'after_pid' must be an integer") from None
        if spec.after_pid <= 0:
            raise ConfigError(f"{where}: 'after_pid' must be positive")

    if raw.get("gpu_free_gb") is not None:
        try:
            spec.gpu_free_gb = float(raw["gpu_free_gb"])
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: 'gpu_free_gb' must be a number (GiB)") from None
        if spec.gpu_free_gb <= 0:
            raise ConfigError(f"{where}: 'gpu_free_gb' must be positive")

    for key, cast, check in (("gpus", int, lambda v: v >= 1),
                             ("stable_for", float, lambda v: v >= 0),
                             ("poll", float, lambda v: v > 0),
                             ("max_retries", int, lambda v: v >= 0)):
        if raw.get(key) is not None:
            try:
                value = cast(raw[key])
            except (TypeError, ValueError):
                raise ConfigError(f"{where}: '{key}' must be a number") from None
            if not check(value):
                raise ConfigError(f"{where}: '{key}' out of range: {raw[key]!r}")
            setattr(spec, key, value)

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

    on_oom = str(raw.get("on_oom", "requeue"))
    if on_oom not in ("requeue", "stop"):
        raise ConfigError(f"{where}: 'on_oom' must be 'requeue' or 'stop', got {on_oom!r}")
    spec.on_oom = on_oom

    if spec.manages_gpu and spec.gpu_index is not None and len(spec.gpu_index) < spec.gpus:
        raise ConfigError(f"{where}: 'gpu_index' lists {len(spec.gpu_index)} gpu(s) "
                          f"but 'gpus' is {spec.gpus}")

    return spec


def load_plan(path: Path, cli_vars: dict[str, str], cli_cwd: Path | None,
              cli_wait_pid: int | None, cli_gpu_free: float | None,
              no_wait: bool) -> Plan:
    """读 task list，展开变量，返回可执行的计划。"""
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML\n{exc}") from exc

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
    variables.update(cli_vars)          # -v 覆盖文件里的 vars

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
        tasks.append(Task(name=name, cmd=substitute(cmd, variables, where)))

    if cli_cwd is not None:
        cwd = cli_cwd
    elif doc.get("cwd"):
        cwd = Path(substitute(str(doc["cwd"]), variables, f"{path}: cwd"))
    else:
        cwd = path.parent            # 默认相对 task list 自身，而不是你 cd 到哪
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ConfigError(f"{path}: cwd not found: {cwd}")

    wait = WaitSpec() if no_wait else _parse_wait(doc.get("wait"), f"{path}: wait")
    if not no_wait:
        if cli_wait_pid is not None:
            wait.after_pid = cli_wait_pid
        if cli_gpu_free is not None:
            wait.gpu_free_gb = cli_gpu_free

    run_name = str(doc.get("name") or path.stem)
    return Plan(tasks=tasks, cwd=cwd, name=run_name, wait=wait)


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


# --------------------------------------------------------------------------- #
# 等：进程 + 显存
# --------------------------------------------------------------------------- #

def pid_cmdline(pid: int) -> str | None:
    """读 /proc/<pid>/cmdline；进程不存在返回 None。"""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return raw.replace(b"\x00", b" ").decode(errors="replace").strip()


def pid_alive(pid: int) -> bool:
    """进程还在跑吗。僵尸态算已经结束——它只是还没被父进程收尸。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    # stat 是 "pid (comm) state ..."，comm 里可能有空格和括号，从最后一个 ')' 切
    try:
        state = stat[stat.rindex(")") + 1:].split()[0]
    except (ValueError, IndexError):
        return True
    return state != "Z"


def query_gpus() -> list[Gpu]:
    """问 nvidia-smi 要每张卡的显存。一次约 40ms。"""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True)
    except FileNotFoundError:
        raise ConfigError("nvidia-smi not found — cannot use 'gpu_free_gb'") from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(f"nvidia-smi failed: {exc}") from exc

    gpus: list[Gpu] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    if not gpus:
        raise ConfigError("nvidia-smi reported no GPUs")
    return gpus


def format_gpus(gpus: list[Gpu]) -> str:
    return "  ".join(f"gpu{g.index}: {g.free_gib:.1f}/{g.total_mib / MIB_PER_GIB:.1f} GiB free"
                     for g in gpus)


class WaitAborted(Exception):
    """等待期间被 Ctrl-C 或超时。"""


def wait_for_pid(pid: int, st: Style, log: LogWriter, poll: float,
                 heartbeat: float = 600.0) -> None:
    """阻塞到 pid 结束。

    注意：pid 不是 tm 的子进程，拿不到它的退出码——只能知道它结束了，
    不知道它是正常收尾还是崩了。要卡这个，在 tasks 里加一步检查产物的守卫任务。
    """
    cmdline = pid_cmdline(pid)
    if cmdline is None or not pid_alive(pid):
        print(st.yellow(f"wait: pid {pid} is not running, moving on"))
        log.write_text(f"# wait: pid {pid} is not running\n")
        return

    print(st.bold(f"wait: holding until pid {pid} exits"))
    print(st.dim(f"      {cmdline}"))
    print(st.dim(f"      its exit code is not observable (not a child of tm) — "
                 f"guard the steps that depend on it"))
    log.write_text(f"# wait: holding until pid {pid} exits\n#   {cmdline}\n")

    started = time.monotonic()
    last_beat = started
    try:
        while pid_alive(pid):
            time.sleep(min(poll, 10.0))
            now = time.monotonic()
            if now - last_beat >= heartbeat:
                last_beat = now
                waited = fmt_duration(now - started)
                print(st.dim(f"      still waiting on {pid} ... ({waited})"))
                log.write_text(f"# wait: still waiting on {pid} ({waited})\n")
    except KeyboardInterrupt:
        raise WaitAborted(f"interrupted while waiting on pid {pid}") from None

    waited = fmt_duration(time.monotonic() - started)
    print(st.green(f"wait: pid {pid} exited (waited {waited})"))
    log.write_text(f"# wait: pid {pid} exited after {waited}\n")


def wait_for_gpus(spec: WaitSpec, st: Style, log: LogWriter,
                  heartbeat: float = 600.0) -> list[int]:
    """阻塞到有 spec.gpus 张卡各自空闲显存 >= spec.gpu_free_gb，返回物理卡号。

    条件必须**连续满足** stable_for 秒才算数。别人的任务刚启动时还没建显存池，
    这时候 nvidia-smi 看着卡是空的，冲进去两边一起 OOM。
    """
    need = spec.gpu_free_gb
    allowed = f"gpu {spec.gpu_index}" if spec.gpu_index is not None else "any gpu"
    print(st.bold(f"wait: need {spec.gpus} x {need:.1f} GiB free on {allowed}, "
                  f"stable for {fmt_duration(spec.stable_for)}"))
    log.write_text(f"# wait: need {spec.gpus} x {need:.1f} GiB free on {allowed}, "
                   f"stable for {spec.stable_for:.0f}s\n")

    started = time.monotonic()
    last_beat = started
    stable_ids: tuple[int, ...] | None = None
    stable_since = 0.0

    try:
        while True:
            gpus = query_gpus()
            cand = [g for g in gpus
                    if (spec.gpu_index is None or g.index in spec.gpu_index) and g.free_gib >= need]
            cand.sort(key=lambda g: -g.free_gib)
            now = time.monotonic()

            if len(cand) >= spec.gpus:
                ids = tuple(sorted(g.index for g in cand[:spec.gpus]))
                if ids != stable_ids:
                    stable_ids, stable_since = ids, now
                    picked = ", ".join(f"gpu{i}" for i in ids)
                    print(st.dim(f"      {picked} look free, confirming for "
                                 f"{fmt_duration(spec.stable_for)} ..."))
                elif now - stable_since >= spec.stable_for:
                    picked = ", ".join(f"gpu{i}" for i in ids)
                    waited = fmt_duration(now - started)
                    print(st.green(f"wait: taking {picked} (waited {waited})"))
                    print(st.dim(f"      {format_gpus(gpus)}"))
                    log.write_text(f"# wait: taking {picked} after {waited}\n"
                                   f"#   {format_gpus(gpus)}\n")
                    return list(ids)
            else:
                if stable_ids is not None:
                    print(st.dim("      no longer free, restarting the countdown"))
                stable_ids, stable_since = None, 0.0

            if spec.timeout is not None and now - started >= spec.timeout:
                raise WaitAborted(f"timed out after {fmt_duration(now - started)} "
                                  f"waiting for {spec.gpus} x {need:.1f} GiB")

            if now - last_beat >= heartbeat:
                last_beat = now
                print(st.dim(f"      waiting ({fmt_duration(now - started)}) — "
                             f"{format_gpus(gpus)}"))
                log.write_text(f"# wait: {fmt_duration(now - started)} — {format_gpus(gpus)}\n")

            time.sleep(spec.poll)
    except KeyboardInterrupt:
        raise WaitAborted("interrupted while waiting for gpus") from None


# --------------------------------------------------------------------------- #
# 输出：终端保原样，日志洗干净
# --------------------------------------------------------------------------- #

class LogWriter:
    """按行落盘，剥掉 ANSI，并把 `\\r` 覆写只保留每行最终状态。

    进度条在终端里是靠 `\\r` 原地刷新的，直接写进文件会变成几万行垃圾。
    同时留一份当前任务输出末尾的环形缓冲，失败时回放给用户看。
    """

    def __init__(self, fh):
        self.fh = fh
        self.buf = b""
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)

    def start_task(self) -> None:
        self.tail.clear()

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk.replace(b"\r\n", b"\n")
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self._write_line(line)

    def flush(self) -> None:
        if self.buf:
            self._write_line(self.buf)
            self.buf = b""
        self.fh.flush()

    def write_text(self, text: str) -> None:
        self.flush()
        self.fh.write(text.encode())
        self.fh.flush()

    def _write_line(self, line: bytes) -> None:
        line = line.rsplit(b"\r", 1)[-1]       # 同一行被覆写过，只留最后一次
        clean = _ANSI_RE.sub(b"", line)
        self.fh.write(clean + b"\n")
        self.tail.append(clean.decode(errors="replace"))


def _set_winsize(fd: int) -> None:
    """把真实终端的尺寸同步给 pty，否则子进程会按 80x24 排版。"""
    try:
        size = os.get_terminal_size()
    except OSError:
        return
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", size.lines, size.columns, 0, 0))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #

def group_members(pgid: int) -> list[int]:
    """还活着的进程组成员。

    孤儿被 init 收养后 ppid 变成 1，从 tm 顺着父子关系找不到它们；
    但 setsid 建的 pgid 不会变，所以按组捞是可靠的。
    """
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == pgid and pid_alive(pid):
                found.append(pid)
        except OSError:      # 进程刚退出，或者不是我们能看的（别人的进程）
            continue
    return sorted(found)


def cleanup_group(pgid: int, st: Style, log: LogWriter) -> None:
    """任务结束后把它那个进程组清干净。

    OOM 之后 dataloader worker、DDP 的其他 rank 经常活下来继续占着显存，
    主进程一退它们就变孤儿，得手动 kill。这个组里只可能有 tm 自己起的东西
    （每个任务 setsid 建独立的组），所以无条件杀是安全的，不用去问 nvidia-smi
    谁占了显存，也就没有误杀别人任务的可能。

    调用时机很关键：leader 必须还是僵尸态（没被 wait() 收尸），pid 才不会被复用，
    pgid 才一定还指着我们自己那个组。
    """
    leftovers = group_members(pgid)
    if not leftovers:
        return

    print(st.yellow(f"    {len(leftovers)} leftover process(es) still alive, killing:"))
    log.write_text(f"# cleanup: {len(leftovers)} leftover process(es) in pgid {pgid}\n")
    for pid in leftovers[:8]:
        desc = pid_cmdline(pid) or "?"
        print(st.dim(f"      {pid}  {desc[:100]}"))
        log.write_text(f"#   {pid}  {desc}\n")
    if len(leftovers) > 8:
        print(st.dim(f"      ... and {len(leftovers) - 8} more"))

    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not group_members(pgid):
                log.write_text(f"# cleanup: group {pgid} is clear\n")
                return
            time.sleep(0.1)

    still = group_members(pgid)
    if still:
        print(st.red(f"    warning: {len(still)} process(es) survived SIGKILL: {still}"))
        log.write_text(f"# cleanup: SURVIVED SIGKILL: {still}\n")


def run_command(cmd: str, cwd: Path, env: dict[str, str], log: LogWriter,
                use_pty: bool, st: Style) -> int:
    """跑一条命令，输出同时送到终端和日志。返回退出码；被 Ctrl-C 打断时返回 130。"""
    out = sys.stdout.buffer
    log.start_task()

    if use_pty:
        # 给子进程一个伪终端，Lightning / tqdm 才会认为自己在交互式终端里
        master, slave = pty.openpty()
        _set_winsize(master)
        popen_kwargs = dict(stdin=slave, stdout=slave, stderr=slave)
    else:
        master = slave = None
        popen_kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    # setsid：每个任务自己一个 session/进程组，pgid == leader 的 pid。
    # 这样任务结束后可以整组杀掉，把 OOM 之后活下来占着显存的 worker 一并带走。
    # 代价是终端的 Ctrl-C 不再自动广播到子进程，得由 tm 转发（见下面的 _terminate）。
    proc = subprocess.Popen(cmd, shell=True, executable=SHELL, cwd=str(cwd), env=env,
                            close_fds=True, start_new_session=True, **popen_kwargs)
    pgid = proc.pid

    if slave is not None:
        os.close(slave)
    read_fd = master if master is not None else proc.stdout.fileno()

    interrupted = False
    try:
        while True:
            try:
                ready, _, _ = select.select([read_fd], [], [], 0.2)
            except InterruptedError:
                continue
            if ready:
                try:
                    chunk = os.read(read_fd, 65536)
                except OSError as exc:
                    if exc.errno == errno.EIO:   # pty 对端关闭，等价于 EOF
                        break
                    raise
                if not chunk:
                    break
                out.write(chunk)
                out.flush()
                log.feed(chunk)
            # 这里不用 proc.poll()：它会顺手收尸，pid 一释放就可能被复用，
            # 后面按 pgid 杀就不保险了。pid_alive() 读 /proc，僵尸态算已结束。
            elif not pid_alive(proc.pid):
                break
    except KeyboardInterrupt:
        interrupted = True
        _terminate(proc, pgid)
    finally:
        if master is not None:
            os.close(master)
        elif proc.stdout is not None:
            proc.stdout.close()
        log.flush()

    # pty 的 EOF 只说明写端都关了，不代表进程退了，先等它真的结束再清理
    while pid_alive(proc.pid):
        time.sleep(0.05)

    # leader 此刻还是僵尸，pid 被占着，pgid 一定还是我们自己那个组
    cleanup_group(pgid, st, log)
    rc = proc.wait()
    return 130 if interrupted else rc


def _terminate(proc: subprocess.Popen, pgid: int) -> None:
    """Ctrl-C 之后确保子进程真的死掉。

    子进程 setsid 出去了，终端不会再把 Ctrl-C 广播给它，得由 tm 转发。
    发给整个组，这样 DDP 的其他 rank、dataloader worker 也一起收到。
    """
    for sig, grace in ((signal.SIGINT, 5.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        if not pid_alive(proc.pid):
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not pid_alive(proc.pid):
                return
            time.sleep(0.1)


@dataclass
class RunResult:
    failed_task: Task | None = None
    failed_index: int = 0
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed_task is None


def run_tasks(tasks: list[Task], cwd: Path, env: dict[str, str], gpu_value: str,
              log: LogWriter, st: Style, use_pty: bool) -> RunResult:
    total = len(tasks)
    result = RunResult()

    for i, task in enumerate(tasks, start=1):
        if not result.ok:
            task.status = "skipped"
            continue

        cmd = unescape_braces(task.cmd.replace("{GPU}", gpu_value))
        task.resolved = cmd
        banner = f"[{i}/{total}] {task.name}"
        print()
        print(st.cyan(st.bold(f"==> {banner}")) + f"  {st.dim(cmd)}")
        log.write_text(f"\n{'=' * 70}\n==> {banner}  ({datetime.now():%H:%M:%S})\n"
                       f"$ {cmd}\n{'=' * 70}\n")

        t0 = time.monotonic()
        rc = run_command(cmd, cwd, env, log, use_pty=use_pty, st=st)
        task.seconds = time.monotonic() - t0
        task.returncode = rc

        if rc == 0:
            task.status = "ok"
            print(st.green(f"<== {task.name} ok ({fmt_duration(task.seconds)})"))
        else:
            task.status = "interrupted" if rc == 130 else "failed"
            result = RunResult(failed_task=task, failed_index=i, tail=list(log.tail))

        log.write_text(f"<== {task.name} {task.status} rc={rc} "
                       f"({fmt_duration(task.seconds)})\n")

    return result


def looks_like_lost_race(result: RunResult, oom_window: float = 600.0) -> bool:
    """这次失败像不像"卡被抢走了"。

    三条同时成立才算：是第一个任务、启动后很快就挂了、输出里有明确的显存不足。
    跑到第 3 个 epoch 才 OOM 那是真问题，重抢没有意义。
    """
    task = result.failed_task
    if task is None or task.status != "failed" or result.failed_index != 1:
        return False
    if task.seconds > oom_window:
        return False
    return any(_OOM_RE.search(line) for line in result.tail)


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #

class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

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
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_failure(result: RunResult, total_steps: int, cwd: Path, gpus: list[int],
                  log_path: Path, st: Style) -> None:
    """失败时把现场完整摆出来：哪一步、什么命令、退出码、输出末尾、日志在哪。"""
    task = result.failed_task
    print()
    print(st.red(st.bold("=" * 70)))
    print(st.red(st.bold(f"FAILED at step {result.failed_index}/{total_steps}: {task.name}")))
    print(st.red(st.bold("=" * 70)))
    print(f"  exit code : {task.returncode}")
    print(f"  duration  : {fmt_duration(task.seconds)}")
    print(f"  cwd       : {cwd}")
    if gpus:
        print(f"  gpu       : {', '.join(str(g) for g in gpus)} "
              f"(CUDA_VISIBLE_DEVICES)")
    print(f"  command   : {task.resolved or task.cmd}")
    if result.tail:
        print()
        print(st.dim(f"  --- last {len(result.tail)} lines of output " + "-" * 38))
        for line in result.tail:
            print(st.dim("  | ") + line)
        print(st.dim("  " + "-" * 66))
    print()
    print(f"  full log  : {log_path}")
    print(st.dim(f"  (grep -n '^==>' {log_path}  列出每步起点)"))


def print_summary(tasks: list[Task], total: float, log_path: Path, st: Style) -> None:
    marks = {
        "ok":          st.green("  ok    "),
        "failed":      st.red("  FAILED"),
        "interrupted": st.yellow("  INTR  "),
        "skipped":     st.dim("  skip  "),
        "pending":     st.dim("  --    "),
    }
    width = max((len(t.name) for t in tasks), default=4)
    print()
    print(st.bold("=" * 60))
    print(st.bold(f"Summary  ({fmt_duration(total)} total)"))
    for t in tasks:
        rc = "" if t.returncode in (0, None) else st.dim(f"  rc={t.returncode}")
        dur = fmt_duration(t.seconds) if t.status in ("ok", "failed", "interrupted") else ""
        print(f"{marks[t.status]}  {t.name:<{width}}  {st.dim(dur):>10}{rc}")
    print(st.dim(f"log: {log_path}"))
    print(st.bold("=" * 60))


def describe_wait(spec: WaitSpec, st: Style) -> list[str]:
    lines: list[str] = []
    if spec.after_pid is not None:
        cmdline = pid_cmdline(spec.after_pid)
        state = cmdline if cmdline else st.yellow("not running — would not wait")
        lines.append(f"  {st.cyan('[wait]')} after pid {spec.after_pid}  {st.dim(state)}")
    if spec.manages_gpu:
        where = f"gpu {spec.gpu_index}" if spec.gpu_index is not None else "any gpu"
        lines.append(f"  {st.cyan('[wait]')} {spec.gpus} x {spec.gpu_free_gb:.1f} GiB free on "
                     f"{where}, stable for {fmt_duration(spec.stable_for)}, "
                     f"poll {fmt_duration(spec.poll)}")
        try:
            lines.append(f"          {st.dim('now: ' + format_gpus(query_gpus()))}")
        except ConfigError as exc:
            lines.append(f"          {st.yellow(str(exc))}")
    return lines


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tm",
        description="Run the commands in a task list one after another; "
                    "stop at the first failure. Can wait for a free GPU first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  tm                                  # run ./{DEFAULT_TASKFILE}\n"
            "  tm --config lists/ccfm_c.yaml\n"
            "  tm --config lists/ccfm_c.yaml --dry-run\n"
            "  tm --config lists/ccfm_c.yaml --gpu-free 50 --wait-pid 243146\n"
        ),
    )
    p.add_argument("-c", "--config", type=Path, default=Path(DEFAULT_TASKFILE),
                   metavar="YAML", help=f"Task list to run (default: ./{DEFAULT_TASKFILE})")
    p.add_argument("-v", "--var", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a value from 'vars:'; repeatable")
    p.add_argument("-C", "--cwd", type=Path, default=None,
                   help="Override 'cwd:' from the task list")
    p.add_argument("-w", "--wait-pid", type=int, default=None, metavar="PID",
                   help="Override wait.after_pid")
    p.add_argument("-g", "--gpu-free", type=float, default=None, metavar="GiB",
                   help="Override wait.gpu_free_gb")
    p.add_argument("--no-wait", action="store_true",
                   help="Ignore the 'wait:' block and start immediately")
    p.add_argument("-n", "--name", default=None,
                   help="Run name, used in the log filename")
    p.add_argument("--log", type=Path, default=None,
                   help=f"Log file path (default: {DEFAULT_LOG_DIR}/<timestamp>_<name>.log)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved plan and exit")
    p.add_argument("--no-device-check", action="store_true",
                   help="Skip the trainer.devices check on the configs referenced by tasks")
    p.add_argument("--no-pty", action="store_true",
                   help="Do not allocate a pty (disables progress bars; use if output looks odd)")
    args = p.parse_args(argv)

    st = Style(sys.stdout.isatty())

    if not args.config.is_file():
        print(st.red(f"error: task list not found: {args.config}"), file=sys.stderr)
        if args.config.name == DEFAULT_TASKFILE:
            print("  create one, or point tm at it:  tm --config path/to/list.yaml",
                  file=sys.stderr)
        return 2

    cli_vars: dict[str, str] = {}
    for item in args.var:
        if "=" not in item:
            p.error(f"--var expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        cli_vars[key.strip()] = value

    try:
        plan = load_plan(args.config, cli_vars, args.cwd, args.wait_pid,
                         args.gpu_free, args.no_wait)
    except ConfigError as exc:
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2

    tasks, cwd, spec = plan.tasks, plan.cwd, plan.wait
    total_steps = len(tasks)

    # tm 要接管显卡的话，先把 config 里的 devices 查掉——
    # 否则可能等了 6 小时才发现 config 写着一张不存在的卡
    device_problems: list[str] = []
    if spec.manages_gpu and not args.no_device_check:
        device_problems = check_device_settings(tasks, cwd, spec.gpus)

    if args.dry_run:
        print(st.bold(f"{total_steps} command(s), cwd={cwd}"))
        for line in describe_wait(spec, st):
            print(line)
        for i, t in enumerate(tasks, start=1):
            gpu_hint = "<gpu>" if "{GPU}" in t.cmd else ""
            print(f"  {st.cyan(f'[{i}/{total_steps}] {t.name}')}  "
                  f"{unescape_braces(t.cmd.replace('{GPU}', gpu_hint))}")
        if device_problems:
            print()
            print(st.red(st.bold("device check failed:")))
            for msg in device_problems:
                print(st.red(f"  - {msg}"))
            return 2
        return 0

    if device_problems:
        print(st.red(st.bold("error: config device check failed")), file=sys.stderr)
        for msg in device_problems:
            print(st.red(f"  - {msg}"), file=sys.stderr)
        print(st.dim("  tm sets CUDA_VISIBLE_DEVICES to the physical card it picks, so the "
                     "job must ask for the relative index."), file=sys.stderr)
        print(st.dim("  fix the config, or pass --no-device-check to skip this."),
              file=sys.stderr)
        return 2

    run_name = args.name or plan.name
    if args.log:
        log_path = args.log.expanduser()
    else:
        log_path = DEFAULT_LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    attempt = 0
    result = RunResult()
    chosen: list[int] = []

    with open(log_path, "wb") as fh:
        log = LogWriter(fh)
        log.write_text(
            f"# tm run '{run_name}' started {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"# task list: {args.config.resolve()}\n"
            f"# cwd: {cwd}\n"
            f"# {total_steps} command(s)\n"
        )
        print(st.bold(f"tm: {total_steps} command(s) from {args.config}, cwd={cwd}"))
        print(st.dim(f"log: {log_path}"))

        while True:
            attempt += 1
            for task in tasks:
                task.reset()

            if spec.active:
                print()
                try:
                    if spec.after_pid is not None:
                        wait_for_pid(spec.after_pid, st, log, spec.poll)
                        spec.after_pid = None      # 重抢时不必再等一次，它已经结束了
                    chosen = wait_for_gpus(spec, st, log) if spec.manages_gpu else []
                except WaitAborted as exc:
                    print(st.yellow(f"\nwait: {exc}"))
                    log.write_text(f"\n# wait aborted: {exc}\n")
                    print(st.yellow("no task was started"))
                    return 1

            env = dict(os.environ, PYTHONUNBUFFERED="1")
            gpu_value = ",".join(str(g) for g in chosen)
            if chosen:
                env["CUDA_VISIBLE_DEVICES"] = gpu_value
                msg = f"using CUDA_VISIBLE_DEVICES={gpu_value}"
                print(st.bold(f"tm: {msg}"))
                log.write_text(f"# {msg}\n")

            result = run_tasks(tasks, cwd, env, gpu_value, log, st, use_pty=not args.no_pty)

            if result.ok or spec.on_oom == "stop" or not spec.manages_gpu:
                break
            if not looks_like_lost_race(result):
                break
            if spec.max_retries and attempt > spec.max_retries:
                print(st.red(f"tm: lost the gpu race {attempt} times, giving up "
                             f"(wait.max_retries = {spec.max_retries})"))
                log.write_text(f"# gave up after {attempt} attempts\n")
                break

            print(st.yellow(f"\ntm: looks like the gpu was taken (attempt {attempt}) "
                            f"— going back to waiting"))
            log.write_text(f"\n# attempt {attempt} lost the gpu race, re-waiting\n")

        total = time.monotonic() - started
        log.write_text(f"\n# finished {datetime.now():%Y-%m-%d %H:%M:%S}, "
                       f"total {fmt_duration(total)}\n")

    if not result.ok:
        if result.failed_task.status == "interrupted":
            print(st.yellow(f"<== {result.failed_task.name} interrupted after "
                            f"{fmt_duration(result.failed_task.seconds)}"))
        else:
            print_failure(result, total_steps, cwd, chosen, log_path, st)

    print_summary(tasks, total, log_path, st)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
