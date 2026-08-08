"""显卡：查询、稳定性判据、按队列顺序的原子分配。

这里是 tm 唯一跨 tick 持有状态的地方（显存采样历史 + 已分配的卡），
而且这些状态**重启后重来一遍反而是对的**——稳定性本来就该重新确认，
已分配的卡则从磁盘上活着的 run 里重建。所以它不需要持久化。

分配规则只有一条，每个 tick 从队列顶往下扫一遍：

    reserved = set()
    for 每个待跑的 list（队列顺序 = 优先级）:
        能用的 = 空闲卡 - 已分配 - reserved
        够 -> 一次拿满，起飞
        不够 -> 把它够得着的空卡塞进 reserved，后面的 list 这个 tick 别想碰

「原子拿满」消灭死锁（不存在持有并等待），reserved 消灭饥饿
（要 2 张卡的 A 不会被后面要 1 张卡的 B 一张张叼光）。
代价是卡会空转着等 A 凑齐——这是有意换来的可预测性。
"""

from __future__ import annotations

import subprocess
import time
from collections import deque
from dataclasses import dataclass, field

from .config import MIB_PER_GIB, ConfigError, WaitSpec

# 显存采样历史保留多久。比任何合理的 stable_for 都长就行。
HISTORY_SECONDS = 1800.0


@dataclass
class Gpu:
    index: int
    total_mib: int
    used_mib: int
    free_mib: int
    util: int = 0

    @property
    def free_gib(self) -> float:
        return self.free_mib / MIB_PER_GIB

    @property
    def total_gib(self) -> float:
        return self.total_mib / MIB_PER_GIB


def query_gpus() -> list[Gpu]:
    """问 nvidia-smi 要每张卡的显存和利用率。一次约 40ms。"""
    try:
        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True)
    except FileNotFoundError:
        raise ConfigError("nvidia-smi not found — cannot use 'gpu_free_gb'") from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(f"nvidia-smi failed: {exc}") from exc

    gpus: list[Gpu] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), int(parts[1]), int(parts[2]),
                            int(parts[3]), int(parts[4])))
        except ValueError:
            continue
    if not gpus:
        raise ConfigError("nvidia-smi reported no GPUs")
    return gpus


@dataclass
class _History:
    """一张卡的空闲显存采样。用来判断「连续 N 秒都够用」。"""
    samples: deque[tuple[float, int]] = field(default_factory=deque)

    def add(self, now: float, free_mib: int) -> None:
        self.samples.append((now, free_mib))
        cutoff = now - HISTORY_SECONDS
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def free_now(self, need_mib: int) -> bool:
        return bool(self.samples) and self.samples[-1][1] >= need_mib

    def stable(self, need_mib: int, window: float, now: float) -> bool:
        """过去 window 秒里每一次采样都 >= need_mib 才算数。

        别人的任务刚启动时还没建显存池，这时候 nvidia-smi 看着卡是空的，
        冲进去两边一起 OOM——所以要看一整段时间，不是看瞬时值。
        """
        if not self.samples:
            return False
        if window <= 0:
            return self.free_now(need_mib)
        # 历史覆盖不到整个窗口（tm 刚起来）就还不能算稳定
        if now - self.samples[0][0] < window:
            return False
        return all(free >= need_mib for ts, free in self.samples if ts >= now - window)


class GpuPool:
    """显存采样 + 分配表。每个 tick 调一次 refresh()，然后按队列顺序 pick()。"""

    def __init__(self):
        self.gpus: list[Gpu] = []
        self.allocated: dict[int, str] = {}      # 卡号 -> 占着它的 run 名
        self._history: dict[int, _History] = {}
        self._reserved: set[int] = set()
        self._now: float = 0.0
        self.error: str | None = None            # 上次 nvidia-smi 出的错

    # ---- 每个 tick ---------------------------------------------------------
    def refresh(self) -> None:
        """采一次样，并开启新一轮扫描（清空 reserved）。"""
        self._now = time.monotonic()
        self._reserved = set()
        try:
            self.gpus = query_gpus()
            self.error = None
        except ConfigError as exc:
            self.error = str(exc)
            return
        for g in self.gpus:
            self._history.setdefault(g.index, _History()).add(self._now, g.free_mib)

    def pick(self, spec: WaitSpec) -> list[int] | None:
        """这个 list 现在能上机吗。能就返回卡号列表，不能就返回 None。

        返回空列表表示「不需要显卡，直接跑」。
        拿不满时会把够得着的空卡记进 reserved，挡住后面优先级更低的 list。
        """
        if not spec.manages_gpu:
            return []
        if self.error:
            return None

        need_mib = int(spec.gpu_free_gb * MIB_PER_GIB)
        eligible = [g for g in self.gpus
                    if (spec.gpu_index is None or g.index in spec.gpu_index)
                    and g.index not in self.allocated]

        ready = [g for g in eligible
                 if g.index not in self._reserved
                 and self._history[g.index].stable(need_mib, spec.stable_for, self._now)]

        if len(ready) >= spec.gpus:
            ready.sort(key=lambda g: -g.free_mib)      # 空得最多的优先
            return sorted(g.index for g in ready[:spec.gpus])

        # 拿不满：一张都不拿（原子），但把够得着的空卡占住不让后面的抢走
        self._reserved |= {g.index for g in eligible
                           if self._history[g.index].free_now(need_mib)}
        return None

    # ---- 分配表 ------------------------------------------------------------
    def allocate(self, indices: list[int], owner: str) -> None:
        for i in indices:
            self.allocated[i] = owner

    def release(self, indices: list[int]) -> None:
        for i in indices:
            self.allocated.pop(i, None)

    def rebuild(self, owned: list[tuple[str, list[int]]]) -> None:
        """从磁盘上还活着的 run 重建分配表。tm 重启后第一件事。"""
        self.allocated = {i: name for name, indices in owned for i in indices}
