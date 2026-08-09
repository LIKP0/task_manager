"""磁盘状态层。

整个 tm 的真相都在这里，内存里没有任何权威状态——tm 随时可以被杀掉重启，
甚至你可以在 tm 没运行的时候直接手改这些文件。

    <仓库目录>/                      默认就是 tm.py 所在的地方，不是 ~/.tm
      lock                          flock 互斥，防止两个终端各跑一个 tm
      paused                        在 = 不扫队列（tm hold / tm resume）
      queue/
        010_ccfm_c.yaml             在这儿 = 还没开始。手动 cp 进来和 `tm add` 等价
        020_ccfm_d.yaml
      runs/
        20260808_143022_ccfm_c/
          list.yaml                 从 queue 移过来的原件
          run.yaml                  tm 写的：状态、抢到的卡、每步的会话名
          01.rc  02.rc              *task 自己写的退出码* —— 唯一的完成凭据
          events.log                人看的流水账

关键的划分：`run.yaml` 是 tm 写的「计划」，`NN.rc` 是 task 写的「结果」。
tm 死了，rc 文件照样在长；rc 文件是什么，tm 说了不算。

放仓库里而不是 ~/.tm：队列和 run 记录是你要经常翻的东西，藏在家目录的点目录里
不好找。代价是它们跟代码同处一个 git 仓库，所以全部进 .gitignore——
运行时状态不该被版本控制，也不该被 `git clean -xdf` 之外的操作碰到。
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

from .config import Plan, list_name, resolve

# 默认把状态放在仓库目录里（tmlib/ 的上一级），不是 ~/.tm。
# 用 __file__ 而不是 cwd：tm 会被 symlink 进 PATH、从任意目录调用，
# .resolve() 会把 symlink 解开，所以指向的始终是真正的仓库。
DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# run 的终态。到了这些状态就不再动它了。
TERMINAL = {"done", "failed", "lost", "timeout", "aborted"}


class LockBusy(Exception):
    """另一个 tm 已经在跑了。"""

    def __init__(self, holder: str):
        super().__init__(holder)
        self.holder = holder


@dataclass
class TaskRecord:
    """run.yaml 里的一步。cmd 是已经展开过变量的最终命令。"""
    name: str
    cmd: str
    session: str = ""
    started: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "cmd": self.cmd,
                "session": self.session, "started": self.started}


class Run:
    """一次 list 的执行。所有读写都落盘，对象本身只是个游标。"""

    def __init__(self, path: Path):
        self.path = path
        self._doc: dict = {}
        self.reload()

    # ---- 读 ---------------------------------------------------------------
    def reload(self) -> None:
        try:
            doc = yaml.safe_load((self.path / "run.yaml").read_text())
        except (OSError, yaml.YAMLError):
            doc = None
        self._doc = doc if isinstance(doc, dict) else {}

    @property
    def name(self) -> str:
        return str(self._doc.get("name") or list_name(self.path))

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
        """这个 run 声明要吃多少显存（每张卡）。共享同一张卡时，别人照着这个数扣账。"""
        try:
            return float(self._doc.get("gpu_budget_gb") or 0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def exclusive(self) -> bool:
        # 缺这个键的是老 run（改 exclusive 之前跑的），按独占算才不会把它挤掉
        return bool(self._doc.get("exclusive", True))

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

    def rc_path(self, index: int) -> Path:
        """第 index 步（1-based）的退出码文件。task 自己往这里写。"""
        return self.path / f"{index:02d}.rc"

    def script_path(self, index: int) -> Path:
        """第 index 步的包装脚本，也就是 tmux 真正执行的东西。

        留在 run 目录里是有用的：事后能看到当时到底跑了什么，
        复现那一步就是 `bash NN.sh`。
        """
        return self.path / f"{index:02d}.sh"

    def rc(self, index: int) -> int | None:
        """读第 index 步的退出码；还没写出来就返回 None。"""
        try:
            text = self.rc_path(index).read_text().strip()
        except (OSError, ValueError):
            return None
        try:
            return int(text)
        except ValueError:
            # 文件在、内容不是数字。包装脚本是先写 .tmp 再 rename 的，正常路径上
            # 到不了这里；留着是防手改和防怪事，方向仍然是「看不懂就当失败」。
            return 1 if text else None

    def scan(self) -> tuple[int, int | None, int | None]:
        """(成功的步数, 失败的步号, 失败的退出码)。

        完全从 rc 文件数出来，不信 run.yaml —— rc 文件是 task 自己写的，
        tm 死过一轮也不影响它们。失败即停，所以失败那步后面不会再有 rc。
        """
        for i in range(1, len(self.tasks) + 1):
            rc = self.rc(i)
            if rc is None:
                return i - 1, None, None      # 第 i 步在跑，或者还没起
            if rc != 0:
                return i - 1, i, rc
        return len(self.tasks), None, None

    def current_index(self) -> int:
        """正在跑（或该起）的是第几步，1-based。全跑完了就返回 len+1。"""
        return self.scan()[0] + 1

    # ---- 写 ---------------------------------------------------------------
    def save(self, **fields) -> None:
        self._doc.update(fields)
        tmp = self.path / "run.yaml.tmp"
        tmp.write_text(yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True))
        tmp.replace(self.path / "run.yaml")     # 原子替换，不会读到半截文件

    def set_state(self, state: str, **fields) -> None:
        if state in TERMINAL:
            fields.setdefault("finished", _now())
        self.save(state=state, **fields)
        self.event(f"state -> {state}")

    def set_task(self, index: int, **fields) -> None:
        tasks = list(self._doc.get("tasks") or [])
        if 1 <= index <= len(tasks) and isinstance(tasks[index - 1], dict):
            tasks[index - 1].update(fields)
            self.save(tasks=tasks)

    def event(self, text: str) -> None:
        with (self.path / "events.log").open("a") as fh:
            fh.write(f"{_now()}  {text}\n")


class Store:
    """`~/.tm/` 这棵目录树。所有磁盘访问都从这里走，测试时换个 root 就行。"""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get("TM_ROOT") or DEFAULT_ROOT).expanduser()
        self.queue_dir = self.root / "queue"
        self.runs_dir = self.root / "runs"
        self._lock_fh = None

    def ensure(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def check_writable(self) -> None:
        """开跑前确认能写盘。

        rc 文件写不进去的话，跑成功的任务也会被判成失败（无 rc = 死了没报告），
        与其跑到一半才发现，不如现在就炸。
        """
        # ensure() 也得包在里面：根目录建不出来时 mkdir 就抛 PermissionError 了，
        # 漏在外面的话用户看到的是一坨 traceback，而不是下面这句话。
        try:
            self.ensure()
            probe = self.root / ".writable"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(f"{self.root} is not writable: {exc}") from exc

    # ---- 单例锁 -----------------------------------------------------------
    @contextmanager
    def lock(self):
        """进程级互斥。进程一死内核自动释放，不会留下僵尸锁。"""
        self.ensure()
        path = self.root / "lock"
        fh = path.open("a+")
        try:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                fh.seek(0)
                raise LockBusy(fh.read().strip() or "unknown") from None
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid {os.getpid()} since {_now()}\n")
            fh.flush()
            self._lock_fh = fh
            yield
        finally:
            self._lock_fh = None
            try:
                fh.seek(0)
                fh.truncate()
                fh.flush()
            except OSError:
                pass
            fh.close()          # 关闭即释放 flock

    # ---- 暂停 -------------------------------------------------------------
    # 这个文件在，tm 就不再从队列里起新的（在跑的不受影响）。用来给你腾出一段
    # 静止时间去重排 / 删改队列，不用担心改到一半有东西被捡走跑了。
    @property
    def pause_flag(self) -> Path:
        return self.root / "paused"

    def paused(self) -> bool:
        return self.pause_flag.exists()

    # ---- 队列 -------------------------------------------------------------
    def queued(self) -> list[Path]:
        """待跑的 list，按文件名排序 —— 这个顺序就是优先级。"""
        if not self.queue_dir.is_dir():
            return []
        return sorted(p for p in self.queue_dir.iterdir()
                      if p.is_file() and p.suffix in (".yaml", ".yml"))

    def add(self, src: Path, seq: int | None = None) -> Path:
        """把一份 yaml 拷进队列。`tm add` 就是这个，跟你手动 cp 完全等价。"""
        self.ensure()
        if seq is None:
            used = [int(m) for p in self.queued()
                    if (m := p.stem.split("_")[0]).isdigit()]
            seq = (max(used) + 10) // 10 * 10 if used else 10
        dst = self.queue_dir / f"{seq:03d}_{list_name(src)}{src.suffix}"
        n = seq
        while dst.exists():                 # 同名就往后挪，不覆盖
            n += 1
            dst = self.queue_dir / f"{n:03d}_{list_name(src)}{src.suffix}"
        # 先写成 .tmp 再 rename：copy 是有中间态的，tm 的 tick 可能正好读到写了一半的
        # yaml。截断处如果落在 task 边界上，它还是合法 yaml，只是少了几步——那就会
        # 悄悄跑一个残缺的 list。rename 是原子的，队列里要么没有要么是完整的。
        # queued() 按后缀过滤，所以 .tmp 期间它是隐形的。
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    # ---- run 目录 ---------------------------------------------------------
    def claim(self, queue_path: Path, plan: Plan, gpus: list[int]) -> Run:
        """把队列里的 yaml 转成一个 run：建目录、移文件、写 run.yaml。

        move 而不是 copy —— 队列目录里出现的永远只有「还没开始」的，
        所以你随时可以放心地手改它，动不到正在跑的东西。
        """
        self.ensure()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.runs_dir / f"{stamp}_{plan.name}"
        n = 1
        while path.exists():                # 同一秒里连开两个
            n += 1
            path = self.runs_dir / f"{stamp}_{plan.name}.{n}"
        path.mkdir(parents=True)

        shutil.move(str(queue_path), str(path / "list.yaml"))

        # 卡在这一刻就定了，所以命令在这里一次性展开完 —— 存进 run.yaml 的是
        # 最终交给 tmux 的那一行。tm 中途重启后照着它继续跑就行，不用再碰 list.yaml。
        gpu_value = ",".join(str(g) for g in gpus)

        run = Run(path)
        run.save(
            name=plan.name,
            state="running",
            cwd=str(plan.cwd),
            gpus=list(gpus),
            gpu_budget_gb=plan.wait.gpu_free_gb or 0,
            exclusive=plan.wait.exclusive,
            started=_now(),
            finished="",
            tasks=[TaskRecord(name=t.name, cmd=resolve(t.cmd, gpu_value)).to_dict()
                   for t in plan.tasks],
        )
        run.event(f"claimed from {queue_path.name}, gpus={gpus or 'none'}")
        return run

    def runs(self, limit: int | None = None) -> list[Run]:
        """所有 run，新的在前。"""
        if not self.runs_dir.is_dir():
            return []
        paths = sorted((p for p in self.runs_dir.iterdir() if p.is_dir()), reverse=True)
        if limit is not None:
            paths = paths[:limit]
        return [Run(p) for p in paths]

    def active(self) -> list[Run]:
        """还没进终态的 run。正常情况下最多几个（每张卡一个）。"""
        return [r for r in self.runs() if not r.done]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
