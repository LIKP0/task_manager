"""tmux 层：把一个 task 丢进一个 tmux 会话，然后隔着进程边界观察它。

tm 不是 task 的父进程——真正 `wait()` 到退出码的是 tmux pane 里那个 sh。
它把退出码写进一个文件，这就是唯一的跨进程通道：

    ( <cmd> ); rc=$?; echo $rc > NN.rc; [ $rc -eq 0 ] && exit 0; exec bash -i
    └ 你的命令      └ 亲爹拿到的真值  └ 落盘        └ 成功即消失    └ 失败则钉住现场

由此得到一张三态表，tm 每个 tick 查一次：

    会话在 + 无 rc   -> 还在跑
    有 rc            -> 结束了，退出码就是文件内容（失败时会话还留着给你 attach）
    会话没 + 无 rc    -> 死了但没来得及报告（被 kill -9 / OOM killer / 机器重启）

第三态是承重墙：**没有凭据就当失败**。少了它，一个被硬杀的 train
会被读成成功，然后 test 抱着半个 checkpoint 跑下去。
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

RUNNING = "running"
DONE = "done"
LOST = "lost"

# 失败后钉住 pane 用的交互 shell。bash 会读 ~/.bashrc，attach 进去 conda 是可用的。
HOLD_SHELL = "bash" if shutil.which("bash") else "sh"


class TmuxError(RuntimeError):
    pass


@dataclass
class State:
    kind: str                 # running / done / lost
    rc: int | None = None


def _tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if not shutil.which("tmux"):
        raise TmuxError("tmux not found — tm runs every task inside a tmux session")
    res = subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=30)
    if check and res.returncode != 0:
        raise TmuxError(f"tmux {' '.join(args)}: {res.stderr.strip() or res.returncode}")
    return res


def session_name(run_name: str, index: int, task_name: str) -> str:
    """tm-ccfm_c-02-test —— 看一眼就知道该 attach 谁，不用查任何东西。"""
    return f"tm-{run_name}-{index:02d}-{task_name}"


def list_sessions() -> list[str]:
    res = _tmux("list-sessions", "-F", "#{session_name}", check=False)
    if res.returncode != 0:          # 没有 server = 一个会话都没有，不是错误
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


class Session:
    """一个 tmux 会话 = 一个正在跑（或已经死掉）的 task。"""

    def __init__(self, name: str, rc_file: Path):
        self.name = name
        self.rc_file = rc_file

    # target 前面的 '=' 强制精确匹配（tmux 默认是前缀匹配，不加的话
    # tm-x-01-a 会命中 tm-x-01-abc）。但 '=' 只有 session 级命令认，
    # capture-pane / display-message 这类 pane 级命令给了会直接报 can't find pane，
    # 所以分成两种写法：判定存活用精确的，抓屏幕用裸名字。
    @property
    def target(self) -> str:
        return f"={self.name}"

    @property
    def pane_target(self) -> str:
        # 裸名字在会话存在时就是精确命中（tmux 先试精确再试前缀），
        # 而这两个调用点只影响显示，不参与「跑完没有」的判定。
        return self.name

    def launch(self, cmd: str, cwd: Path, env: dict[str, str]) -> None:
        """起会话。命令跑完会把退出码写进 rc_file。"""
        if self.alive():
            raise TmuxError(f"session {self.name} already exists — "
                            f"kill it first: tmux kill-session -t {self.name}")
        # 命令套在子 shell 里跑。不套的话，命令自己写了 `exit`（或者以 exec 收尾）
        # 会把包装 shell 一起带走，rc 文件永远写不出来，tm 就把一次正常的失败
        # 读成了「无凭据 = LOST」。子 shell 把 exit 挡在里面，$? 照样是真值。
        wrapper = (
            f"( {cmd} ); rc=$?; echo $rc > {shlex.quote(str(self.rc_file))}; "
            f"[ $rc -eq 0 ] && exit 0; exec {HOLD_SHELL} -i"
        )
        args = ["new-session", "-d", "-s", self.name, "-c", str(cwd)]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        args.append(wrapper)
        _tmux(*args)

    def alive(self) -> bool:
        return _tmux("has-session", "-t", self.target, check=False).returncode == 0

    def state(self) -> State:
        """三态判定。先看 rc 文件，再看会话——顺序不能反。

        反过来的话会撞上一个窗口：任务刚写完 rc、会话正在消失的那一瞬间，
        先查会话会看到「没了」，再查 rc 却是有的，得多绕一次。
        先查 rc 则永远不会误判。
        """
        rc = self._read_rc()
        if rc is not None:
            return State(DONE, rc)
        if self.alive():
            return State(RUNNING)
        return State(LOST)          # 无凭据 = 失败

    def _read_rc(self) -> int | None:
        try:
            text = self.rc_file.read_text().strip()
        except OSError:
            return None
        if not text:
            return None             # 文件刚建、还没写进去
        try:
            return int(text)
        except ValueError:
            return 1                # 写到一半被打断，当失败

    def silent_for(self) -> float | None:
        """这个 pane 多久没有输出了（秒）。拿不到就返回 None。

        用来给死锁/卡住的任务标黄——只报警，不动手。存 checkpoint、
        跑 CPU 的 eval 都会安静很久，自动 kill 迟早误伤。
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
        """抓当前屏幕内容，给 `tm ls` 显示失败现场的最后几行。"""
        res = _tmux("capture-pane", "-p", "-t", self.pane_target, check=False)
        if res.returncode != 0:
            return []
        out = [ln.rstrip() for ln in res.stdout.splitlines()]
        while out and not out[-1]:
            out.pop()
        return out[-lines:]

    def kill(self) -> None:
        _tmux("kill-session", "-t", self.target, check=False)


def enable_activity_tracking() -> None:
    """#{window_activity} 需要开 monitor-activity 才会更新。

    这是 server 全局选项，副作用仅仅是状态栏会高亮有输出的窗口。
    """
    _tmux("set-option", "-g", "monitor-activity", "on", check=False)
