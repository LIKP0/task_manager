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

「已分配」默认就是独占：卡上有 run 就不再进候选。`exclusive: false` 的 list 才允许共享一张卡，
这时判据有两条，缺一不可：

    实测的 free（挡别人的进程）  和  total - 卡上各 run 声明的额度（挡我们自己的）

第二条不能省。tm 刚把 A 起上去时 A 还在 import torch，nvidia-smi 看着卡是空的，
只看实测值 B 就会挤进来，等两边都建完显存池一起 OOM。而 A 要吃多少 tm 是知道的——
已知的事不该靠采样去猜。
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


@dataclass
class Claim:
    """一个 run 占着一张卡时留下的账。

    owner 是 run 的目录名（唯一），不是 list 名——同名的 list 是允许的，
    共享一张卡时按 list 名销账会连带把另一笔也抹掉。
    budget 是这个 run 声明要吃的显存。
    """
    owner: str
    budget_mib: int
    exclusive: bool = True


class GpuPool:
    """显存采样 + 分配表。每个 tick 调一次 refresh()，然后按队列顺序 pick()。"""

    def __init__(self):
        self.gpus: list[Gpu] = []
        self.claims: dict[int, list[Claim]] = {}   # 卡号 -> 占着它的 run
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
                    and self._can_join(g, need_mib, spec.exclusive)]

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
    def _can_join(self, g: Gpu, need_mib: int, exclusive: bool) -> bool:
        """这张卡容不容得下我。空卡永远容得下。"""
        held = self.claims.get(g.index) or []
        if not held:
            return True
        if exclusive or any(c.exclusive for c in held):
            return False          # 任一方声明独占，整张卡就独占
        # 共享：账面上剩的够不够。不看实测值——那边由 stable() 单独把关
        booked = sum(c.budget_mib for c in held)
        return g.total_mib - booked >= need_mib

    def allocate(self, indices: list[int], owner: str,
                 budget_mib: int = 0, exclusive: bool = True) -> None:
        for i in indices:
            self.claims.setdefault(i, []).append(Claim(owner, budget_mib, exclusive))

    def release(self, indices: list[int], owner: str) -> None:
        """只销 owner 自己那笔账，同卡上别人的留着。"""
        for i in indices:
            held = [c for c in self.claims.get(i, []) if c.owner != owner]
            if held:
                self.claims[i] = held
            else:
                self.claims.pop(i, None)

    def rebuild(self, owned: list[tuple[str, list[int], int, bool]]) -> None:
        """从磁盘上还活着的 run 重建分配表。tm 重启后第一件事。"""
        self.claims = {}
        for name, indices, budget_mib, exclusive in owned:
            self.allocate(indices, name, budget_mib, exclusive)

    def owners(self, index: int) -> list[str]:
        return [c.owner for c in self.claims.get(index, [])]
