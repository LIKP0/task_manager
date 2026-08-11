"""GPUs: querying, the stability test, and atomic allocation in queue order.

This is the only place tm keeps state across ticks (VRAM sample history plus the
allocation table), and rebuilding both after a restart is the correct behaviour
anyway: stability should be re-established from scratch, and the allocation table is
reconstructed from the runs still alive on disk. So none of it needs persisting.

There is one allocation rule, applied top-down over the queue every tick:

    reserved = set()
    for each pending list (queue order = priority):
        available = free cards - allocated - reserved
        enough    -> take them all at once and start
        not enough-> put the free cards it could have used into reserved, so lists
                     below it cannot touch them this tick

Taking all cards at once removes deadlock (there is no hold-and-wait), and reserved
removes starvation (a list wanting 2 cards is not picked apart by a later list
wanting 1). The cost is cards idling while the bigger list assembles its set —
predictability bought on purpose.

Allocation is exclusive by default: a card with a run on it leaves the candidate
pool. Only a list with `exclusive: false` may share, and then two tests both apply:

    measured free (keeps out other people's processes)
    total - sum of budgets declared by runs on the card (keeps out our own)

The second is not optional. Right after tm starts A, A is still importing torch and
nvidia-smi shows the card as empty; going by the measured value alone lets B in, and
both OOM once their memory pools are built. How much A intends to use is something
tm already knows, and known facts should not be guessed at by sampling.
"""

from __future__ import annotations

import subprocess
import time
from collections import deque
from dataclasses import dataclass, field

from .config import ConfigError, WaitSpec

# MiB is nvidia-smi's unit, so the conversion lives here. Callers speak GiB only.
MIB_PER_GIB = 1024

# How long to keep VRAM samples. Anything longer than a sane stable_for will do.
HISTORY_SECONDS = 1800.0


@dataclass
class Gpu:
    index: int
    total_mib: int
    free_mib: int
    util: int = 0

    @property
    def free_gib(self) -> float:
        return self.free_mib / MIB_PER_GIB

    @property
    def total_gib(self) -> float:
        return self.total_mib / MIB_PER_GIB


def query_gpus() -> list[Gpu]:
    """Ask nvidia-smi for per-card memory and utilisation. About 40ms per call."""
    try:
        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free,utilization.gpu",
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


@dataclass
class _History:
    """Free-VRAM samples for one card, used to test "enough for N seconds running"."""
    samples: deque[tuple[float, int]] = field(default_factory=deque)

    def add(self, now: float, free_mib: int) -> None:
        self.samples.append((now, free_mib))
        cutoff = now - HISTORY_SECONDS
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def free_now(self, need_mib: int) -> bool:
        return bool(self.samples) and self.samples[-1][1] >= need_mib

    def stable(self, need_mib: int, window: float, now: float) -> bool:
        """True only if every sample in the last `window` seconds was >= need_mib.

        A job that just started has not built its memory pool yet, so nvidia-smi
        shows the card as empty; moving in then makes both sides OOM. Hence a whole
        interval rather than an instantaneous reading.
        """
        if not self.samples:
            return False
        if window <= 0:
            return self.free_now(need_mib)
        # History not yet covering the full window (tm just started) is not stable
        if now - self.samples[0][0] < window:
            return False
        return all(free >= need_mib for ts, free in self.samples if ts >= now - window)


# One run's claim on one card: (owner, budget_mib, exclusive).
#
# owner is the run directory name, which is unique, not the list name: duplicate list
# names are allowed, and releasing by list name would wipe someone else's claim on a
# shared card. budget_mib is the VRAM that run declared it needs.
Claim = tuple[str, int, bool]


class GpuPool:
    """VRAM sampling plus the allocation table.

    Call refresh() once per tick, then pick() in queue order.
    """

    def __init__(self):
        self.gpus: list[Gpu] = []
        self.claims: dict[int, list[Claim]] = {}   # card index -> runs holding it
        self._history: dict[int, _History] = {}
        self._reserved: set[int] = set()
        self._now: float = 0.0
        self.error: str | None = None            # last nvidia-smi error, if any

    # ---- per tick ------------------------------------------------------------
    def refresh(self) -> None:
        """Take a sample and begin a new scan (clearing reserved)."""
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
        """Can this list start now? Returns the card indices, or None.

        An empty list means "no GPU needed, run immediately". When the request cannot
        be filled, the free cards it could have used go into reserved, blocking
        lower-priority lists for the rest of this tick.
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
            ready.sort(key=lambda g: -g.free_mib)      # emptiest first
            return sorted(g.index for g in ready[:spec.gpus])

        # Cannot fill the request: take nothing (atomic), but hold the free cards it
        # could have used so lists below cannot take them
        self._reserved |= {g.index for g in eligible
                           if self._history[g.index].free_now(need_mib)}
        return None

    # ---- allocation table ------------------------------------------------------
    def _can_join(self, g: Gpu, need_mib: int, exclusive: bool) -> bool:
        """Does this card have room for me? An empty card always does."""
        held = self.claims.get(g.index) or []
        if not held:
            return True
        if exclusive or any(is_exclusive for _, _, is_exclusive in held):
            return False          # either side claiming exclusive makes it exclusive
        # Sharing: is there budget left on paper? The measured value is stable()'s job
        booked = sum(budget for _, budget, _ in held)
        return g.total_mib - booked >= need_mib

    def allocate(self, indices: list[int], owner: str,
                 budget_gb: float = 0.0, exclusive: bool = True) -> None:
        """Book `budget_gb` per card for `owner`. GiB in, MiB kept internally."""
        budget_mib = int(budget_gb * MIB_PER_GIB)
        for i in indices:
            self.claims.setdefault(i, []).append((owner, budget_mib, exclusive))

    def release(self, indices: list[int], owner: str) -> None:
        """Release only this owner's claim, leaving others on the same card."""
        for i in indices:
            held = [c for c in self.claims.get(i, []) if c[0] != owner]
            if held:
                self.claims[i] = held
            else:
                self.claims.pop(i, None)

    def rebuild(self, owned: list[tuple[str, list[int], float, bool]]) -> None:
        """Rebuild the table from runs still alive on disk. First thing after a restart."""
        self.claims = {}
        for name, indices, budget_gb, exclusive in owned:
            self.allocate(indices, name, budget_gb, exclusive)
