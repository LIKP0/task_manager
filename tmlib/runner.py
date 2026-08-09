"""tmux 层：把一个 task 丢进一个 tmux 会话，然后隔着进程边界观察它。

tm 不是 task 的父进程——真正 `wait()` 到退出码的是 tmux pane 里那个 bash。
它把退出码写进一个文件，这就是唯一的跨进程通道。那段包装不是拼在命令行上的，
而是**落成 run 目录里的 `NN.sh`**，tmux 收到的只有一句 `bash NN.sh`（见 write_script）。

由此得到一张三态表，tm 每个 tick 查一次：

    会话在 + 无 rc   -> 还在跑
    有 rc            -> 结束了，退出码就是文件内容（失败时会话还留着给你 attach）
    会话没 + 无 rc    -> 死了但没来得及报告（被 kill -9 / OOM killer / 机器重启）

第三态是承重墙：**没有凭据就当失败**。少了它，一个被硬杀的 train
会被读成成功，然后 test 抱着半个 checkpoint 跑下去。

反过来「假成功」是不允许存在的，所以 rc 的取值路径要经得起推敲：
写不进盘 -> 无 rc -> 当失败；被硬杀 -> 无 rc -> 当失败；rc 文件用 rename 落地，
不存在读到半截。唯一能骗过它的是命令自己把失败吞掉（管道、后台化），
所以包装脚本开了 `pipefail`——见 write_script 里的注释。
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

# 跑包装脚本的 shell。必须是 bash：`set -o pipefail` 在 dash/sh 里没有，
# 而没有它，`train.py | tee log` 里 train 崩了也会得到 rc=0（实测）。
# 没装 bash 就退回 sh，同时不写 pipefail——功能降级，但不会语法错误。
BASH = shutil.which("bash")
SCRIPT_SHELL = BASH or "/bin/sh"

# 失败后钉住 pane 用的交互 shell。bash 会读 ~/.bashrc，attach 进去 conda 是可用的。
HOLD_SHELL = "bash" if BASH else "sh"


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


def write_script(path: Path, cmd: str, cwd: Path, rc_file: Path, label: str) -> None:
    """把一步任务写成一个自足的 bash 脚本，这就是 tmux 真正执行的东西。

    落成文件而不是拼在 tmux 命令行上，买到四件事：
      1. shell 无关——tmux 用 default-shell 执行命令串，不能假设那是 POSIX shell
      2. 能开 `set -o pipefail`，也能安全地写多行，不用跟嵌套引号搏斗
      3. 事后能看到当时到底跑了什么，复现失败就是 `bash NN.sh`
      4. 跟 rc 文件、队列目录一样，状态摊在文件系统上，没有只存在于内存里的东西
    """
    rc_tmp = Path(f"{rc_file}.tmp")
    lines = [
        f"#!{SCRIPT_SHELL}",
        f"# tm: {label}",
        "# 这就是 tm 交给 tmux 跑的全部内容。可以直接执行它来复现这一步。",
    ]
    if BASH:
        # 管道的退出码默认只看最后一个命令：`python train.py | tee log` 里
        # train 崩了、tee 成功，$? 依然是 0，于是 tm 把失败读成成功、
        # 抱着半个 checkpoint 往下跑 test。pipefail 把它掰回来。
        lines.append("set -o pipefail")
    lines += [
        # 脚本自己 cd，所以脱离 tm 单独执行也是对的（tmux 的 -c 只管 pane 的初始 cwd）
        f"cd {shlex.quote(str(cwd))} || exit 1",
        "",
        # 命令套在子 shell 里跑。不套的话，命令自己写了 `exit`（或者以 exec 收尾）
        # 会把包装一起带走，rc 文件永远写不出来，tm 就把一次正常的失败
        # 读成了「无凭据 = LOST」。子 shell 把 exit 挡在里面，$? 照样是真值。
        f"( {cmd} )",
        "rc=$?",
        "",
        # 先写临时文件再 rename。rename 是原子的，所以 rc 文件要么不存在、
        # 要么内容完整——「读到半截」这类问题从「已处理」变成「不可能发生」。
        # 写不进去时 && 短路，不留下 rc 文件，tm 判 LOST（当失败），方向是安全的。
        f"echo $rc > {shlex.quote(str(rc_tmp))} && "
        f"mv {shlex.quote(str(rc_tmp))} {shlex.quote(str(rc_file))}",
        "",
        # 成功就自己消失（不留垃圾会话），失败就钉在原地：
        # pane 保留完整 scrollback，attach 进去是个站在 job cwd 和环境里的交互 shell。
        "[ $rc -eq 0 ] && exit 0",
        f"exec {HOLD_SHELL} -i",
        "",
    ]
    path.write_text("\n".join(lines))
    path.chmod(0o755)


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

    def launch(self, cmd: str, cwd: Path, env: dict[str, str], script: Path) -> None:
        """起会话。命令跑完会把退出码写进 rc_file。

        `script` 是包装脚本的落点（run 目录里的 `NN.sh`）。
        """
        if self.alive():
            raise TmuxError(f"session {self.name} already exists — "
                            f"kill it first: tmux kill-session -t {self.name}")
        write_script(script, cmd, cwd, self.rc_file, self.name)
        args = ["new-session", "-d", "-s", self.name, "-c", str(cwd)]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        # 交给 tmux 的就这一句。tmux 是拿 default-shell（默认 $SHELL）来 -c 执行它的，
        # 而 `bash <path>` 在 fish / csh / zsh 里都是合法的一条命令调用——
        # 把包装的 shell 语法关在文件里，就不用假设用户的登录 shell 是 POSIX 的。
        args.append(f"{SCRIPT_SHELL} {shlex.quote(str(script))}")
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
