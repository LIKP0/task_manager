#!/usr/bin/env python3
# 注意：pane 只从 tm 继承 PATH，其余环境变量取自 tmux server 首次启动时的环境（实测 3.2a）。
# 所以 python 找得对，但 conda activate 才有的 CONDA_PREFIX / LD_LIBRARY_PATH 会丢。
"""tm —— 把 task list 排进队列，等到卡就自动上机。

每个 task 跑在自己的 tmux 会话里，tm 只做三件事：按顺序扫队列、看谁能上机、
盯着 rc 文件等结果。tm 自己不持有任何权威状态——全在 ~/.tm/ 下面，
所以它随时可以被杀掉重启，也可以在它没运行的时候手改队列。

    tm                    消费 ~/.tm/queue/ 里的 task list（建议挂在 tmux 里）
    tm ls                 看进度。tm 在不在跑都能用
    tm add list.yaml      加进队列（就是 cp，你手动拷也一样）
    tm check list.yaml    只解析不跑，看看展开成什么样
    tm attach [name]      attach 到正在跑的那个 task
    tm clean              清掉失败留下的 tmux 会话

设计上刻意不做的事：不自动重试、不自动 kill 卡住的任务、不做资源公平调度。
调度权在你手里——队列顺序就是优先级，tm 只负责照着执行。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tmlib import runner
from tmlib.config import ConfigError, check_device_settings, load_plan, unescape_braces
from tmlib.gpu import GpuPool
from tmlib.store import LockBusy, Store
from tmlib.view import Style, render

POLL_SECONDS = 10.0


# --------------------------------------------------------------------------- #
# tm run —— 主循环
# --------------------------------------------------------------------------- #

def cmd_run(store: Store, args, st: Style) -> int:
    try:
        store.check_writable()
    except RuntimeError as exc:
        # rc 文件写不进去的话，跑成功的任务也会被判成失败，不如现在就炸
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2

    try:
        with store.lock():
            return _loop(store, args, st)
    except LockBusy as exc:
        print(st.red(f"error: another tm is already running ({exc.holder})"), file=sys.stderr)
        print(st.dim("  看它在干什么：tm ls"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(st.yellow("\ntm: 停止调度。已经起来的 task 不受影响，还在各自的 tmux 里跑着。"))
        print(st.dim("  tm ls 看进度；重新 tm 会接着往下推。"))
        return 130


def _loop(store: Store, args, st: Style) -> int:
    runner.enable_activity_tracking()
    pool = GpuPool()
    waiting_since: dict[Path, float] = {}
    complained: set[Path] = set()

    print(st.bold(f"tm: watching {store.queue_dir}"))
    idle_announced = False

    while True:
        pool.refresh()
        # 分配表每个 tick 从磁盘重建：tm 重启后不会把别人占着的卡再发一次
        pool.rebuild([(r.name, r.gpus) for r in store.active()])

        for run in store.active():
            _advance(run, pool, st)

        _start_pending(store, pool, st, args, waiting_since, complained)

        active, queued = store.active(), store.queued()
        if not active and not queued:
            if args.once:
                print(st.green("tm: 队列空了，退出。"))
                return 0
            if not idle_announced:
                print(st.dim(f"tm: 队列空了，待命中。放东西进 {store.queue_dir} 就会自动开跑。"))
                idle_announced = True
        else:
            idle_announced = False

        time.sleep(args.poll)


def _advance(run, pool: GpuPool, st: Style) -> None:
    """推进一个 run：把跑完的收掉，把下一步起起来。

    进度完全由 rc 文件决定（`run.scan()`），不看 run.yaml —— 所以上一步一旦写出
    rc=0，这次调用就会直接把下一步起起来，中间不多隔一个 tick。
    """
    run.reload()
    done, fail_i, fail_rc = run.scan()
    tasks = run.tasks

    if fail_i is not None:
        rec = tasks[fail_i - 1]
        run.set_state("failed", failed_step=fail_i, failed_rc=fail_rc)
        pool.release(run.gpus)
        print(st.red(f"\n<== {run.name} FAILED at step {fail_i}/{len(tasks)} "
                     f"({rec.name}) rc={fail_rc}"))
        _print_tail(rec, run, fail_i, st)
        print(st.dim(f"    现场还在：tmux attach -t {rec.session}"))
        print(st.dim(f"    {run.path}"))
        return

    if done >= len(tasks):
        run.set_state("done")
        pool.release(run.gpus)
        print(st.green(f"\n<== {run.name} done ({len(tasks)}/{len(tasks)})"))
        return

    idx = done + 1
    rec = tasks[idx - 1]
    sess = runner.Session(runner.session_name(run.name, idx, rec.name), run.rc_path(idx))

    if not rec.started:
        _launch(run, idx, rec, sess, st)
        return

    if sess.state().kind == runner.LOST:
        # 无 rc 文件、会话也没了：被 kill -9 / OOM killer / 机器重启带走的。
        # 没有凭据就当失败——绝不能当成功往下跑。
        run.set_state("lost", failed_step=idx)
        pool.release(run.gpus)
        print(st.red(f"\n<== {run.name} LOST at step {idx}/{len(tasks)} ({rec.name})"))
        print(st.dim("    会话消失且没有留下退出码——被硬杀或机器重启了"))


def _print_tail(rec, run, index: int, st: Style, lines: int = 12) -> None:
    """失败时把 pane 里最后几行摆出来，省得为了看一眼报错还得 attach。"""
    sess = runner.Session(rec.session, run.rc_path(index))
    tail = sess.capture(lines)
    if not tail:
        return
    print(st.dim(f"    --- {rec.session} 最后 {len(tail)} 行 " + "-" * 30))
    for line in tail:
        print(st.dim("    | ") + line)


def _launch(run, idx: int, rec, sess: runner.Session, st: Style) -> None:
    env = {"PYTHONUNBUFFERED": "1"}
    if run.gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in run.gpus)
    try:
        sess.launch(rec.cmd, Path(run.cwd), env, run.script_path(idx))
    except (runner.TmuxError, OSError) as exc:
        run.set_state("aborted", note=str(exc))
        print(st.red(f"tm: cannot launch {run.name} step {idx}: {exc}"))
        return
    run.set_task(idx, session=sess.name, started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run.event(f"step {idx} {rec.name} -> {sess.name}")
    print(st.cyan(st.bold(f"\n==> {run.name} [{idx}] {rec.name}")) + st.dim(f"  {rec.cmd}"))
    print(st.dim(f"    tmux attach -t {sess.name}"))


def _start_pending(store: Store, pool: GpuPool, st: Style, args,
                   waiting_since: dict[Path, float], complained: set[Path]) -> None:
    """按队列顺序扫一遍，能上机的就上。

    顺序就是优先级：拿不满卡的 list 会把它够得着的空卡 reserve 住，
    排在它后面的这个 tick 别想碰——不然要两张卡的永远排不上。
    """
    now = time.monotonic()
    for path in store.queued():
        try:
            plan = load_plan(path, cli_vars=args.vars)
        except ConfigError as exc:
            if path not in complained:
                complained.add(path)
                print(st.red(f"tm: {path.name} 读不了，跳过：{exc}"))
            continue
        complained.discard(path)

        gpus = pool.pick(plan.wait)
        if gpus is None:
            first = waiting_since.setdefault(path, now)
            if plan.wait.timeout is not None and now - first >= plan.wait.timeout:
                run = store.claim(path, plan, [])
                run.set_state("timeout")
                waiting_since.pop(path, None)
                print(st.yellow(f"tm: {plan.name} 等卡超时（{plan.wait.timeout:.0f}s），跳过"))
            continue

        if plan.wait.manages_gpu and not args.no_device_check:
            problems = check_device_settings(plan.tasks, plan.cwd, plan.wait.gpus)
            if problems:
                # 别让它在队列里一遍遍重试——挪出去，把问题说清楚
                run = store.claim(path, plan, [])
                run.set_state("aborted", note="device check failed")
                print(st.red(f"tm: {plan.name} config device check 不通过，跳过："))
                for msg in problems:
                    print(st.red(f"  - {msg}"))
                print(st.dim("  tm 用 CUDA_VISIBLE_DEVICES 指卡，所以 config 里要写相对编号。"))
                continue

        run = store.claim(path, plan, gpus)
        pool.allocate(gpus, plan.name)
        waiting_since.pop(path, None)
        where = ",".join(f"gpu{g}" for g in gpus) or "no gpu"
        print(st.bold(f"\ntm: start {plan.name} on {where}  ({len(plan.tasks)} tasks)"))
        _advance(run, pool, st)


# --------------------------------------------------------------------------- #
# 其它子命令
# --------------------------------------------------------------------------- #

def cmd_ls(store: Store, args, st: Style) -> int:
    for line in render(store, st, recent=args.recent):
        print(line)
    return 0


def cmd_add(store: Store, args, st: Style) -> int:
    rc = 0
    for src in args.files:
        if not src.is_file():
            print(st.red(f"error: not found: {src}"), file=sys.stderr)
            rc = 2
            continue
        try:
            # 带上 -v：队列里的 yaml 是 run 的时候用当时 tm 的 -v 重新解析的，
            # 这里不传的话，add 这一关会比现实更严，把能跑的 list 拦在门外。
            load_plan(src, cli_vars=args.vars)  # 先验一遍，别把坏 yaml 放进队列
        except ConfigError as exc:
            print(st.red(f"error: {src}: {exc}"), file=sys.stderr)
            rc = 2
            continue
        dst = store.add(src, seq=args.seq)
        print(f"queued  {st.cyan(dst.name)}")
    if rc == 0:
        print(st.dim(f"（顺序就是文件名顺序，改名即可调整；rm 掉就是取消）"))
    return rc


def cmd_check(store: Store, args, st: Style) -> int:
    try:
        plan = load_plan(args.file, cli_vars=args.vars)
    except ConfigError as exc:
        print(st.red(f"error: {exc}"), file=sys.stderr)
        return 2
    spec = plan.wait
    print(st.bold(f"{plan.name}: {len(plan.tasks)} tasks, cwd={plan.cwd}"))
    if spec.manages_gpu:
        where = ("gpu " + ",".join(map(str, spec.gpu_index))
                 if spec.gpu_index is not None else "any gpu")
        print(st.cyan("  [wait] ") + f"{spec.gpus} x {spec.gpu_free_gb:.1f} GiB on {where}, "
              f"stable for {spec.stable_for:.0f}s")
    else:
        print(st.cyan("  [wait] ") + "no gpu requirement — starts immediately")
    for i, t in enumerate(plan.tasks, start=1):
        # 显示前展开 `{{` / `}}`，否则 check 给你看的和 shell 真正收到的不是一回事。
        # `{GPU}` 留着不动——卡要等 claim 那一刻才定，这里还不知道。
        print(f"  {st.cyan(f'[{i}] {t.name}')}  {unescape_braces(t.cmd)}")
    if spec.manages_gpu:
        problems = check_device_settings(plan.tasks, plan.cwd, spec.gpus)
        for msg in problems:
            print(st.red(f"  ! {msg}"))
        if problems:
            return 2
    return 0


def cmd_attach(store: Store, args, st: Style) -> int:
    live = set(runner.list_sessions())
    targets = []
    for run in store.active():
        idx = run.current_index()
        tasks = run.tasks
        if idx <= len(tasks) and tasks[idx - 1].session in live:
            targets.append(tasks[idx - 1].session)
    if args.name:
        targets = [s for s in live if args.name in s] or [args.name]
    if not targets:
        print(st.yellow("没有在跑的 task。"), file=sys.stderr)
        return 1
    if len(targets) > 1:
        print(st.bold("有多个，指定一个："))
        for name in targets:
            print(f"  tm attach {name}")
        return 1
    os.execvp("tmux", ["tmux", "attach", "-t", f"={targets[0]}"])
    return 0                                   # execvp 不会返回


def cmd_clean(store: Store, args, st: Style) -> int:
    """清掉失败留下的 pane。在跑的一律不动。"""
    keep = set()
    for run in store.active():
        keep.update(t.session for t in run.tasks if t.session)
    victims = [s for s in runner.list_sessions() if s.startswith("tm-") and s not in keep]
    if not victims:
        print(st.dim("没有可清理的会话。"))
        return 0
    for name in victims:
        if not args.yes:
            print(f"  would kill  {name}")
        else:
            runner.Session(name, Path("/nonexistent")).kill()
            print(f"  killed  {name}")
    if not args.yes:
        print(st.dim(f"加 -y 真的执行（{len(victims)} 个）"))
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def _parse_vars(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--var expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tm", description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="队列在 ~/.tm/queue/，顺序就是优先级。手动 cp 一个 yaml 进去等价于 tm add。")
    p.add_argument("--root", type=Path, default=None, help="覆盖 ~/.tm")
    p.add_argument("-v", "--var", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖 yaml 里 vars: 的值，可重复")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="消费队列（默认）")
    run.add_argument("--poll", type=float, default=POLL_SECONDS, metavar="SEC")
    run.add_argument("--once", action="store_true", help="队列跑空就退出，不待命")
    run.add_argument("--no-device-check", action="store_true")

    ls = sub.add_parser("ls", help="看进度（tm 没在跑也能用）")
    ls.add_argument("-n", "--recent", type=int, default=5, help="显示最近几个跑完的")

    add = sub.add_parser("add", help="把 yaml 加进队列")
    add.add_argument("files", nargs="+", type=Path)
    add.add_argument("--seq", type=int, default=None, help="指定排序号，默认排到最后")

    check = sub.add_parser("check", help="只解析不跑")
    check.add_argument("file", type=Path)

    attach = sub.add_parser("attach", help="attach 到正在跑的 task")
    attach.add_argument("name", nargs="?", default=None)

    clean = sub.add_parser("clean", help="清掉失败留下的 tmux 会话")
    clean.add_argument("-y", "--yes", action="store_true", help="真的执行")

    args = p.parse_args(argv)
    args.vars = _parse_vars(args.var)
    st = Style()
    store = Store(args.root)

    handlers = {None: cmd_run, "run": cmd_run, "ls": cmd_ls, "add": cmd_add,
                "check": cmd_check, "attach": cmd_attach, "clean": cmd_clean}
    if args.cmd in (None, "run"):
        for name, default in (("poll", POLL_SECONDS), ("once", False),
                              ("no_device_check", False)):
            if not hasattr(args, name):
                setattr(args, name, default)
    return handlers[args.cmd](store, args, st)


if __name__ == "__main__":
    sys.exit(main())
