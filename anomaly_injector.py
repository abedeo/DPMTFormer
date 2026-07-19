from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


EPS = 1e-6


@dataclass(frozen=True)
class SpikeConfig:
    prob: float = 0.35
    count_min: int = 1
    count_max: int = 4
    width_min: int = 1
    width_max: int = 2
    amp_k_min: float = 1.2
    amp_k_max: float = 2.5


@dataclass(frozen=True)
class OscillationConfig:
    prob: float = 0.35
    dur_min: int = 6
    dur_max: int = 12
    amp_k_min: float = 1.2
    amp_k_max: float = 2.5
    period_min: int = 2
    period_max: int = 6
    noise_ratio: float = 0.12


@dataclass(frozen=True)
class RampConfig:
    prob: float = 0.30
    dur_min: int = 8
    dur_max: int = 12
    amp_k_min: float = 1.2
    amp_k_max: float = 2.5
    shape_prob_step: float = 0.4


@dataclass(frozen=True)
class AnomalyInjectConfig:
    enabled: bool = True
    identity_prob: float = 0.40
    point_ratio_min: float = 0.02
    point_ratio_max: float = 0.05

    local_scale_window: int = 36
    min_local_scale: float = 0.02

    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    max_events_per_window: int = 3
    max_tries_per_event: int = 30

    spike: SpikeConfig = SpikeConfig()
    oscillation: OscillationConfig = OscillationConfig()
    ramp: RampConfig = RampConfig()


def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x))


def _local_scale(x: np.ndarray, center: int, half_win: int, floor: float) -> float:
    n = x.shape[0]
    l = max(0, center - half_win)
    r = min(n, center + half_win + 1)
    s = _safe_std(x[l:r])
    if not np.isfinite(s):
        s = 0.0
    return max(float(s), float(floor))


def _clip_inplace(x: np.ndarray, clip_min: Optional[float], clip_max: Optional[float]) -> None:
    if clip_min is not None or clip_max is not None:
        np.clip(x, clip_min, clip_max, out=x)


def _choose_type(rng: np.random.Generator, cfg: AnomalyInjectConfig) -> str:
    p_spk = max(0.0, float(cfg.spike.prob))
    p_osc = max(0.0, float(cfg.oscillation.prob))
    p_rmp = max(0.0, float(cfg.ramp.prob))
    s = p_spk + p_osc + p_rmp
    if s <= EPS:
        return "spike"
    p = np.array([p_spk, p_osc, p_rmp], dtype=np.float64)
    p = p / p.sum()
    t = int(rng.choice(3, p=p))
    return ["spike", "oscillation", "ramp"][t]


def inject_spike(
    x_noisy: np.ndarray,
    x_clean: np.ndarray,
    mask: np.ndarray,
    cfg: AnomalyInjectConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = x_noisy.shape[0]
    c = cfg.spike

    count = int(rng.integers(c.count_min, c.count_max + 1))
    touched_idx: List[int] = []

    for _ in range(count):
        pos = int(rng.integers(0, n))
        width = int(rng.integers(c.width_min, c.width_max + 1))
        end = min(n, pos + width)

        scale = _local_scale(
            x_clean, center=pos, half_win=cfg.local_scale_window // 2, floor=cfg.min_local_scale
        )
        k = float(rng.uniform(c.amp_k_min, c.amp_k_max))
        sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        amp = sign * k * scale

        x_noisy[pos:end] = x_noisy[pos:end] + amp
        mask[pos:end] = True
        touched_idx.extend(list(range(pos, end)))

    _clip_inplace(x_noisy, cfg.clip_min, cfg.clip_max)

    return {
        "type": "spike",
        "count": count,
        "points": int(len(set(touched_idx))),
        "idx_min": int(min(touched_idx)) if touched_idx else -1,
        "idx_max": int(max(touched_idx)) if touched_idx else -1,
    }


def inject_oscillation(
    x_noisy: np.ndarray,
    x_clean: np.ndarray,
    mask: np.ndarray,
    cfg: AnomalyInjectConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = x_noisy.shape[0]
    c = cfg.oscillation

    dur = int(rng.integers(c.dur_min, c.dur_max + 1))
    if dur >= n:
        dur = n - 1
    start = int(rng.integers(0, max(1, n - dur)))
    end = start + dur

    center = (start + end) // 2
    scale = _local_scale(
        x_clean, center=center, half_win=cfg.local_scale_window // 2, floor=cfg.min_local_scale
    )
    k = float(rng.uniform(c.amp_k_min, c.amp_k_max))
    amp = k * scale

    period = int(rng.integers(c.period_min, c.period_max + 1))
    t = np.arange(dur, dtype=np.float32)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    wave = amp * np.sin((2.0 * np.pi / max(1, period)) * t + phase)
    noise = rng.normal(loc=0.0, scale=max(EPS, c.noise_ratio * amp), size=dur).astype(np.float32)
    delta = wave.astype(np.float32) + noise

    x_noisy[start:end] = x_noisy[start:end] + delta
    mask[start:end] = True

    _clip_inplace(x_noisy, cfg.clip_min, cfg.clip_max)

    return {
        "type": "oscillation",
        "start": int(start),
        "end": int(end - 1),
        "dur": int(dur),
        "amp": float(amp),
        "period": int(period),
    }


def inject_ramp_or_step(
    x_noisy: np.ndarray,
    x_clean: np.ndarray,
    mask: np.ndarray,
    cfg: AnomalyInjectConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = x_noisy.shape[0]
    c = cfg.ramp

    dur = int(rng.integers(c.dur_min, c.dur_max + 1))
    if dur >= n:
        dur = n - 1
    start = int(rng.integers(0, max(1, n - dur)))
    end = start + dur

    center = (start + end) // 2
    scale = _local_scale(
        x_clean, center=center, half_win=cfg.local_scale_window // 2, floor=cfg.min_local_scale
    )

    k = float(rng.uniform(c.amp_k_min, c.amp_k_max))
    sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    peak = sign * k * scale

    use_step = float(rng.random()) < float(c.shape_prob_step)
    if use_step:
        delta = np.full((dur,), peak, dtype=np.float32)
        shape_name = "step"
    else:
        up_len = max(1, dur // 2)
        down_len = dur - up_len
        up = np.linspace(0.0, peak, num=up_len, endpoint=False, dtype=np.float32)
        down = np.linspace(peak, 0.0, num=max(1, down_len), endpoint=True, dtype=np.float32)
        delta = np.concatenate([up, down], axis=0)[:dur]
        shape_name = "ramp"

    x_noisy[start:end] = x_noisy[start:end] + delta
    mask[start:end] = True

    _clip_inplace(x_noisy, cfg.clip_min, cfg.clip_max)

    return {
        "type": "ramp",
        "shape": shape_name,
        "start": int(start),
        "end": int(end - 1),
        "dur": int(dur),
        "peak": float(peak),
    }


class AnomalyInjector:
    def __init__(self, config: Optional[AnomalyInjectConfig] = None, seed: int = 2026) -> None:
        self.cfg = config if config is not None else AnomalyInjectConfig()
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        x: np.ndarray,
        force_identity: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x_clean = np.asarray(x, dtype=np.float32).copy()
        n = x_clean.shape[0]

        if n <= 1:
            return x_clean.copy(), np.zeros_like(x_clean, dtype=bool), {
                "is_identity": True,
                "events": [],
                "target_ratio": 0.0,
                "actual_ratio": 0.0,
            }

        if not self.cfg.enabled:
            return x_clean.copy(), np.zeros((n,), dtype=bool), {
                "is_identity": True,
                "events": [],
                "target_ratio": 0.0,
                "actual_ratio": 0.0,
            }

        if force_identity is None:
            is_identity = bool(self.rng.random() < self.cfg.identity_prob)
        else:
            is_identity = bool(force_identity)

        if is_identity:
            return x_clean.copy(), np.zeros((n,), dtype=bool), {
                "is_identity": True,
                "events": [],
                "target_ratio": 0.0,
                "actual_ratio": 0.0,
            }

        x_noisy = x_clean.copy()
        mask = np.zeros((n,), dtype=bool)
        events: List[Dict[str, Any]] = []

        target_ratio = float(self.rng.uniform(self.cfg.point_ratio_min, self.cfg.point_ratio_max))
        target_points = max(1, int(round(target_ratio * n)))

        for _ in range(max(1, self.cfg.max_events_per_window)):
            if int(mask.sum()) >= target_points:
                break

            ok = False
            for _ in range(max(1, self.cfg.max_tries_per_event)):
                t = _choose_type(self.rng, self.cfg)
                before = int(mask.sum())

                if t == "spike":
                    e = inject_spike(x_noisy, x_clean, mask, self.cfg, self.rng)
                elif t == "oscillation":
                    e = inject_oscillation(x_noisy, x_clean, mask, self.cfg, self.rng)
                else:
                    e = inject_ramp_or_step(x_noisy, x_clean, mask, self.cfg, self.rng)

                after = int(mask.sum())
                if after > before:
                    events.append(e)
                    ok = True
                    break

            if not ok:
                break

        actual_ratio = float(mask.mean()) if n > 0 else 0.0
        meta: Dict[str, Any] = {
            "is_identity": False,
            "events": events,
            "target_ratio": float(target_ratio),
            "actual_ratio": actual_ratio,
            "n_events": int(len(events)),
            "n_points": int(mask.sum()),
        }
        return x_noisy, mask, meta


def build_default_injector(seed: int = 2026) -> AnomalyInjector:
    return AnomalyInjector(config=AnomalyInjectConfig(), seed=seed)