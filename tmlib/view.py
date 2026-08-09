"""给人看的输出。`tm ls` 的全部内容都是现读磁盘 + tmux 拼出来的，
所以 tm 在不在跑都能用——这正是把状态放盘上换来的。
"""

from __future__ import annotations

import fcntl
import sys
from datetime import datetime

from . import runner
from .config import ConfigError, load_plan
from .gpu import query_gpus
from .store import Store

# 静默多久开始标黄。存 checkpoint、跑 CPU 的 eval 都会安静很久，别设太短。
SILENT_WARN = 30 * 60.0


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
        t0 = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(until, "%Y-%m-%d %H:%M:%S") if until else datetime.now()
    except ValueError:
        return ""
    return fmt_duration((t1 - t0).total_seconds())


def render(store: Store, st: Style, recent: int = 5) -> list[str]:
    """`tm ls` 的完整输出。"""
    out: list[str] = []
    sessions = set(runner.list_sessions())

    holder = _lock_holder(store)
    out.append(st.bold("tm: ") + (st.green(holder) if holder else st.dim("not running")))

    # ---- 在跑的 ------------------------------------------------------------
    active = store.active()
    if active:
        out.append("")
        out.append(st.bold("RUNNING"))
    for run in active:
        done, fail_i, fail_rc = run.scan()
        tasks = run.tasks
        idx = done + 1
        gpu = ",".join(f"gpu{g}" for g in run.gpus) or "-"
        step = tasks[idx - 1].name if idx <= len(tasks) else "-"
        line = (f"  {run.name:<14} {gpu:<8} [{idx}/{len(tasks)}] {step:<12} "
                f"{_elapsed(run.started):>7}")

        if idx <= len(tasks) and tasks[idx - 1].session:
            sess = runner.Session(tasks[idx - 1].session, run.rc_path(idx))
            if sess.name in sessions:
                silent = sess.silent_for()
                if silent is not None and silent >= SILENT_WARN:
                    line += st.yellow(f"  ⚠ silent {fmt_duration(silent)}")
                line += st.dim(f"  -> tmux attach -t {sess.name}")
            else:
                line += st.red("  session gone")
        out.append(line)

    # ---- 排队的 ------------------------------------------------------------
    queued = store.queued()
    if queued:
        out.append("")
        out.append(st.bold("QUEUED") + st.dim("   (顺序 = 优先级，改文件名即可调整)"))
    for path in queued:
        try:
            plan = load_plan(path)
            spec = plan.wait
            where = ("gpu " + ",".join(map(str, spec.gpu_index))
                     if spec.gpu_index is not None else "any gpu")
            note = (f"{spec.gpus}x{spec.gpu_free_gb:.0f}GiB on {where}"
                    if spec.manages_gpu else "no gpu needed")
            out.append(f"  {path.name:<24} {len(plan.tasks)} tasks   {st.dim(note)}")
        except ConfigError as exc:
            out.append(f"  {path.name:<24} {st.red('BAD: ' + str(exc).splitlines()[0])}")

    # ---- 跑完的 ------------------------------------------------------------
    finished = [r for r in store.runs(limit=recent + len(active)) if r.done][:recent]
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

    # ---- 显卡 --------------------------------------------------------------
    try:
        gpus = query_gpus()
        out.append("")
        out.append(st.bold("GPUS"))
        for g in gpus:
            bar = st.red if g.free_gib < 5 else (st.yellow if g.util > 50 else st.green)
            out.append(f"  gpu{g.index}: {bar(f'{g.free_gib:6.1f}')}/{g.total_gib:.1f} GiB free"
                       f"   util {g.util:3d}%")
    except ConfigError as exc:
        out.append("")
        out.append(st.yellow(f"GPUS: {exc}"))

    if not active and not queued:
        out.append("")
        out.append(st.dim(f"队列是空的。放个 yaml 进去：cp list.yaml {store.queue_dir}/"))
    return out


def _lock_holder(store: Store) -> str:
    """谁拿着锁。空 = 没人在跑。

    判据是「能不能非阻塞地抢到锁」而不是文件内容：tm 被硬杀时内核只回收 flock，
    那行字会留在文件里，光读就会指着一个早没了的 pid 说它还在跑。
    """
    path = store.root / "lock"
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
                return text                    # 抢不到 = 真的有 tm 拿着
            fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        return text                            # 连打开都不行，只能信文件内容
    return ""                                  # 抢得到 = 上一个 tm 是被硬杀的
