from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from anomaly_injector import AnomalyInjector, AnomalyInjectConfig
from loader import DATE_COL, LABEL_COL, RAIN_COL, TIME_COL, WATER_COL


STEPS_PER_HOUR = 12
HISTORY_HOURS = 12
HORIZON_HOURS = 3
TOTAL_HOURS = HISTORY_HOURS + HORIZON_HOURS

HIST_STEPS = HISTORY_HOURS * STEPS_PER_HOUR
FUT_STEPS = HORIZON_HOURS * STEPS_PER_HOUR
TOTAL_STEPS = TOTAL_HOURS * STEPS_PER_HOUR


PRED_PATCH_SIZE = STEPS_PER_HOUR
N_PATCH = TOTAL_STEPS // PRED_PATCH_SIZE
N_HIST_PATCH = HIST_STEPS // PRED_PATCH_SIZE
N_FUT_PATCH = FUT_STEPS // PRED_PATCH_SIZE


RECON_PATCH_SIZE = 6
N_RECON_PATCH = TOTAL_STEPS // RECON_PATCH_SIZE
N_RECON_HIST_PATCH = HIST_STEPS // RECON_PATCH_SIZE


COV_DIM = 5
EPS = 1e-6


@dataclass(frozen=True)
class GlobalNormStats:
    water_mean: float
    water_std: float
    rain_hour_logsum_mean: float
    rain_hour_logsum_std: float


def _to_float_np(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float32)


def _valid_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, valid in enumerate(mask.astype(bool, copy=False)):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _local_sigma_mask(
    water: np.ndarray,
    window_len: int,
    sigma_k: float,
    single_sigma_k: float,
    min_std: float,
) -> np.ndarray:
    window_len = max(3, int(window_len))
    min_neighbors = max(3, min(window_len - 1, window_len // 2))
    out = np.zeros(water.shape[0], dtype=bool)

    for start, end in _valid_segments(np.isfinite(water)):
        segment = water[start:end].astype(np.float64, copy=False)
        if segment.size <= min_neighbors:
            continue

        left_width = window_len // 2
        right_width = window_len - left_width - 1
        padded = np.full(segment.size + left_width + right_width, np.nan, dtype=np.float64)
        padded[left_width : left_width + segment.size] = segment
        neighbors = np.lib.stride_tricks.sliding_window_view(padded, window_len).copy()
        neighbors[:, left_width] = np.nan

        count = np.isfinite(neighbors).sum(axis=1)
        center = np.nanmedian(neighbors, axis=1)
        mad = np.nanmedian(np.abs(neighbors - center[:, None]), axis=1)
        scale = np.maximum(1.4826 * mad, float(min_std))
        z = np.divide(
            np.abs(segment - center),
            scale,
            out=np.zeros_like(segment),
            where=(count >= min_neighbors) & np.isfinite(center) & np.isfinite(scale),
        )

        moderate = z >= float(sigma_k)
        severe = z >= float(single_sigma_k)
        persistent = np.zeros_like(moderate)
        for run_start, run_end in _valid_segments(moderate):
            if run_end - run_start >= 2:
                persistent[run_start:run_end] = True
        out[start:end] = np.logical_or(persistent, severe)

    return out


def _time_features(date01: np.ndarray, time01: np.ndarray) -> np.ndarray:
    tod = (time01 % 1.0) * (2.0 * np.pi)
    doy = (date01 % 1.0) * (2.0 * np.pi)
    return np.stack([np.sin(tod), np.cos(tod), np.sin(doy), np.cos(doy)], axis=-1).astype(np.float32)


def _patchify_1d(x: np.ndarray, patch_size: int) -> np.ndarray:
    assert x.ndim == 1
    assert x.shape[0] % patch_size == 0
    return x.reshape(x.shape[0] // patch_size, patch_size)


def _patchify_2d(x: np.ndarray, patch_size: int) -> np.ndarray:
    assert x.ndim == 2
    assert x.shape[0] % patch_size == 0
    return x.reshape(x.shape[0] // patch_size, patch_size, x.shape[1])


def _instance_stats_wg_history(
    w_g_hist: np.ndarray,
    visible_mask: np.ndarray,
    sigma_floor: float,
) -> Tuple[float, float, float, float]:
    assert w_g_hist.shape[0] == HIST_STEPS
    assert visible_mask.shape[0] == HIST_STEPS

    idx = np.where(visible_mask)[0]
    if idx.size == 0:
        mu = 0.0
        log_sigma_clamped = float(np.log(max(sigma_floor, 1e-8)))
        trend_per_hour = 0.0
        log_raw_sigma = float(np.log(1e-8))
        return mu, log_sigma_clamped, trend_per_hour, log_raw_sigma

    x = w_g_hist[idx]
    mu = float(np.mean(x))

    raw_sigma = float(np.std(x))
    raw_sigma_safe = max(raw_sigma, 1e-8)
    log_raw_sigma = float(np.log(raw_sigma_safe))

    sigma = max(raw_sigma, float(sigma_floor))
    log_sigma_clamped = float(np.log(sigma))

    if idx.size < 2:
        trend_per_hour = 0.0
    else:
        t = idx.astype(np.float32)
        t = t - t.mean()
        denom = float(np.sum(t * t)) + EPS
        slope_per_step = float(np.sum(t * (x - x.mean())) / denom)
        trend_per_hour = slope_per_step * STEPS_PER_HOUR

    return mu, log_sigma_clamped, trend_per_hour, log_raw_sigma


def compute_global_norm_stats(
    split_data: Dict[str, dict],
    clip_quantiles: Optional[Tuple[float, float]] = (0.005, 0.995),
) -> GlobalNormStats:
    waters: List[np.ndarray] = []
    rain_hour_logsum: List[np.ndarray] = []

    for _, item in split_data.items():
        df: pd.DataFrame = item["df"]
        segs = item["segments_kept"]

        w_all = _to_float_np(df[WATER_COL])
        r_all = _to_float_np(df[RAIN_COL])

        for seg in segs:
            w = w_all[seg.start : seg.end + 1]
            r = r_all[seg.start : seg.end + 1]

            w = w[np.isfinite(w)]
            if w.size:
                waters.append(w.astype(np.float32))

            r = r[np.isfinite(r)]
            if r.size >= PRED_PATCH_SIZE:
                r = np.maximum(r, 0.0).astype(np.float32)
                u = np.log1p(r)
                n_full = (u.shape[0] // PRED_PATCH_SIZE) * PRED_PATCH_SIZE
                if n_full > 0:
                    u = u[:n_full].reshape(-1, PRED_PATCH_SIZE)
                    U = np.sum(u, axis=1).astype(np.float32)
                    rain_hour_logsum.append(U)

    if not waters:
        raise ValueError("没有可用于计算全局归一化的水位数据。")
    if not rain_hour_logsum:
        rain_hour_logsum = [np.zeros((1,), dtype=np.float32)]

    w_cat = np.concatenate(waters).astype(np.float32)
    r_cat = np.concatenate(rain_hour_logsum).astype(np.float32)

    if clip_quantiles is not None:
        lo_q, hi_q = clip_quantiles
        w_lo, w_hi = np.quantile(w_cat, [lo_q, hi_q])
        r_lo, r_hi = np.quantile(r_cat, [lo_q, hi_q])
        w_cat = np.clip(w_cat, w_lo, w_hi)
        r_cat = np.clip(r_cat, r_lo, r_hi)

    w_mean = float(np.mean(w_cat))
    w_std = max(float(np.std(w_cat)), 1e-4)

    r_mean = float(np.mean(r_cat))
    r_std = max(float(np.std(r_cat)), 1e-4)

    return GlobalNormStats(
        water_mean=w_mean,
        water_std=w_std,
        rain_hour_logsum_mean=r_mean,
        rain_hour_logsum_std=r_std,
    )


class X3TwoPassDataset(Dataset):
    def __init__(
        self,
        split_data: Dict[str, dict],
        global_stats: GlobalNormStats,
        samples_per_epoch: int = 20000,
        seed: int = 2026,
        deterministic_rec_hour: bool = False,
        sigma_floor: float = 0.05,

        station_uniform: bool = True,
        seg_weight_by_len: bool = True,

        event_prob: float = 0.0,
        event_future_hours: int = 3,
        event_dh_thresh: float = 0.5,
        event_max_tries: int = 50,

        use_synth_anomaly: bool = True,
        anomaly_cfg: Optional[AnomalyInjectConfig] = None,
        use_offline_anomaly_label: bool = False,
        local_sigma_enabled: bool = False,
        local_sigma_window: int = 0,
        local_sigma_k: float = 5.0,
        local_sigma_single_k: float = 6.0,
        local_sigma_min_std: float = 1e-6,
    ) -> None:
        self.split_data = split_data
        self.global_stats = global_stats
        self.samples_per_epoch = int(samples_per_epoch)
        self.rng = np.random.default_rng(seed)

        self.deterministic_rec_hour = bool(deterministic_rec_hour)
        self._rec_hour_cursor = 0
        self.sigma_floor = float(sigma_floor)

        self.station_uniform = bool(station_uniform)
        self.seg_weight_by_len = bool(seg_weight_by_len)

        self.event_prob = float(event_prob)
        self.event_future_hours = int(event_future_hours)
        self.event_dh_thresh = float(event_dh_thresh)
        self.event_max_tries = int(event_max_tries)

        self.use_synth_anomaly = bool(use_synth_anomaly)
        self.use_offline_anomaly_label = bool(use_offline_anomaly_label)
        if self.use_synth_anomaly and self.use_offline_anomaly_label:
            raise ValueError("synthetic injection and offline anomaly labels cannot be enabled together")
        self.anomaly_injector = AnomalyInjector(
            config=anomaly_cfg if anomaly_cfg is not None else AnomalyInjectConfig(),
            seed=seed + 1024,
        )
        self.local_sigma_enabled = bool(local_sigma_enabled)
        self.local_sigma_window = int(local_sigma_window)
        self.local_sigma_k = float(local_sigma_k)
        self.local_sigma_single_k = float(local_sigma_single_k)
        self.local_sigma_min_std = float(local_sigma_min_std)
        if self.local_sigma_k <= 0 or self.local_sigma_single_k < self.local_sigma_k:
            raise ValueError("local sigma thresholds must satisfy 0 < local_sigma_k <= local_sigma_single_k")
        self.seed = int(seed)

        if self.event_future_hours <= 0:
            raise ValueError("event_future_hours must > 0")
        self._event_future_steps = self.event_future_hours * STEPS_PER_HOUR

        self._stations: List[str] = []
        self._station_to_segs: Dict[str, List[Any]] = {}
        self._station_seg_weights: Dict[str, Optional[np.ndarray]] = {}

        for station_id, item in split_data.items():
            segs_ok = [seg for seg in item["segments_kept"] if seg.length >= TOTAL_STEPS]
            if not segs_ok:
                continue

            self._stations.append(station_id)
            self._station_to_segs[station_id] = segs_ok

            if self.seg_weight_by_len:
                w = np.array([float(seg.length) for seg in segs_ok], dtype=np.float64)
                w = w / np.sum(w)
                self._station_seg_weights[station_id] = w
            else:
                self._station_seg_weights[station_id] = None

        if not self._stations:
            raise ValueError("没有站点包含可用窗口。")

        self._seg_index: List[Tuple[str, Any]] = []
        for station_id in self._stations:
            for seg in self._station_to_segs[station_id]:
                self._seg_index.append((station_id, seg))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def reset_rng(self, seed: Optional[int] = None) -> None:
        s = self.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(s)
        self._rec_hour_cursor = 0
        if self.use_synth_anomaly:
            self.anomaly_injector = AnomalyInjector(
                config=self.anomaly_injector.cfg,
                seed=s + 1024,
            )

    def set_rec_hour(self, hour_idx: int) -> None:
        if not (0 <= hour_idx < N_HIST_PATCH):
            raise ValueError(f"hour_idx 必须在 [0,{N_HIST_PATCH-1}]")
        self._rec_hour_cursor = int(hour_idx)


    def _sample_station_and_seg(self) -> Tuple[str, Any]:
        if self.station_uniform:
            station_id = self._stations[int(self.rng.integers(0, len(self._stations)))]
            segs = self._station_to_segs[station_id]
            w = self._station_seg_weights.get(station_id, None)
            if w is None:
                seg = segs[int(self.rng.integers(0, len(segs)))]
            else:
                seg = segs[int(self.rng.choice(len(segs), p=w))]
            return station_id, seg

        station_id, seg = self._seg_index[int(self.rng.integers(0, len(self._seg_index)))]
        return station_id, seg

    def _sample_t0_in_seg(self, seg: Any) -> int:
        max_start = seg.end - TOTAL_STEPS + 1
        if max_start < seg.start:
            raise RuntimeError("段长度异常，无法采样窗口。")
        return int(self.rng.integers(seg.start, max_start + 1))

    def _window_is_event(self, w_win: np.ndarray) -> bool:
        if w_win.shape[0] != TOTAL_STEPS:
            return False

        s = HIST_STEPS
        e = min(HIST_STEPS + self._event_future_steps, TOTAL_STEPS)
        if e <= s:
            return False

        w_fut = w_win[s:e]
        m = np.isfinite(w_fut)
        if m.sum() < 2:
            return False

        amp = float(np.nanmax(w_fut) - np.nanmin(w_fut))
        return amp > self.event_dh_thresh

    def _sample_window_normal(self) -> Tuple[str, pd.DataFrame, int]:
        station_id, seg = self._sample_station_and_seg()
        df: pd.DataFrame = self.split_data[station_id]["df"]
        t0 = self._sample_t0_in_seg(seg)
        return station_id, df, t0

    def _sample_window_event(self) -> Tuple[str, pd.DataFrame, int]:
        for _ in range(max(1, self.event_max_tries)):
            station_id, seg = self._sample_station_and_seg()
            df: pd.DataFrame = self.split_data[station_id]["df"]
            t0 = self._sample_t0_in_seg(seg)

            w_win = _to_float_np(df[WATER_COL])[t0 : t0 + TOTAL_STEPS]
            if self._window_is_event(w_win):
                return station_id, df, t0

        return self._sample_window_normal()

    def _sample_window(self) -> Tuple[str, pd.DataFrame, int]:
        if self.event_prob > 0 and float(self.rng.random()) < self.event_prob:
            return self._sample_window_event()
        return self._sample_window_normal()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        station_id, df, t0 = self._sample_window()

        w_clean = _to_float_np(df[WATER_COL])[t0 : t0 + TOTAL_STEPS]
        r = _to_float_np(df[RAIN_COL])[t0 : t0 + TOTAL_STEPS]
        date01 = _to_float_np(df[DATE_COL])[t0 : t0 + TOTAL_STEPS]
        time01 = _to_float_np(df[TIME_COL])[t0 : t0 + TOTAL_STEPS]

        w_mask = np.isfinite(w_clean)
        if not np.all(w_mask):
            s = pd.Series(w_clean.astype(np.float32))
            if s.notna().sum() == 0:
                w_clean = np.zeros_like(w_clean, dtype=np.float32)
            else:
                w_clean = s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float32)
            w_mask = np.isfinite(w_clean)

        w_raw_clean = w_clean.astype(np.float32)

        if self.use_offline_anomaly_label:
            if LABEL_COL not in df.columns:
                raise ValueError(f"offline anomaly label column is missing: {LABEL_COL}")
            w_raw_noisy = w_raw_clean.copy()
            anom_mask = _to_float_np(df[LABEL_COL])[t0 : t0 + TOTAL_STEPS] > 0.5
            anom_meta = {
                "is_identity": not bool(np.any(anom_mask)),
                "events": [],
                "n_events": int(np.any(anom_mask)),
            }
        elif self.use_synth_anomaly:
            w_raw_noisy, anom_mask, anom_meta = self.anomaly_injector.inject(w_raw_clean)
        else:
            w_raw_noisy = w_raw_clean.copy()
            anom_mask = np.zeros((TOTAL_STEPS,), dtype=bool)
            anom_meta = {"is_identity": True, "events": [], "target_ratio": 0.0, "actual_ratio": 0.0}

        if self.local_sigma_enabled:
            sigma_mask = _local_sigma_mask(
                w_raw_clean,
                window_len=self.local_sigma_window if self.local_sigma_window > 0 else TOTAL_STEPS,
                sigma_k=self.local_sigma_k,
                single_sigma_k=self.local_sigma_single_k,
                min_std=self.local_sigma_min_std,
            )
            anom_mask = np.logical_or(anom_mask, sigma_mask)

        rec_is_identity = not bool(np.any(anom_mask))
        rec_n_events = int(anom_meta.get("n_events", 0))
        if self.local_sigma_enabled:
            rec_n_events += len(_valid_segments(sigma_mask))

        w_g_clean = (w_raw_clean - self.global_stats.water_mean) / (self.global_stats.water_std + EPS)
        w_g_noisy = (w_raw_noisy - self.global_stats.water_mean) / (self.global_stats.water_std + EPS)

        hist_visible = w_mask[:HIST_STEPS]
        mu, log_sigma_clamped, trend_per_hour, log_raw_sigma = _instance_stats_wg_history(
            w_g_noisy[:HIST_STEPS],
            hist_visible,
            sigma_floor=self.sigma_floor,
        )

        sigma = float(np.exp(log_sigma_clamped))
        z_noisy = (w_g_noisy - mu) / (sigma + EPS)
        z_clean = (w_g_clean - mu) / (sigma + EPS)

        tf = _time_features(date01, time01)

        r = np.maximum(r, 0.0).astype(np.float32)
        u = np.log1p(r).astype(np.float32)

        u_patch = _patchify_1d(u, patch_size=PRED_PATCH_SIZE)
        U_hour = np.sum(u_patch, axis=1, keepdims=True).astype(np.float32)
        U_hour_g = (U_hour - self.global_stats.rain_hour_logsum_mean) / (
            self.global_stats.rain_hour_logsum_std + EPS
        )

        z_noisy_patch_pred = _patchify_1d(z_noisy, patch_size=PRED_PATCH_SIZE)
        z_clean_patch_pred = _patchify_1d(z_clean, patch_size=PRED_PATCH_SIZE)
        tf_patch_pred = _patchify_2d(tf, patch_size=PRED_PATCH_SIZE)
        time_mean_pred = np.mean(tf_patch_pred, axis=1).astype(np.float32)

        cov_patch_pred = np.concatenate([U_hour_g, time_mean_pred], axis=-1).astype(np.float32)

        hist_z_patches = z_noisy_patch_pred[:N_HIST_PATCH]
        fut_z_patches = z_clean_patch_pred[N_HIST_PATCH : N_HIST_PATCH + N_FUT_PATCH]

        stats_vec = np.array([mu, log_sigma_clamped, trend_per_hour, log_raw_sigma], dtype=np.float32)

        z_noisy_patch_rec = _patchify_1d(z_noisy, patch_size=RECON_PATCH_SIZE)
        z_clean_patch_rec = _patchify_1d(z_clean, patch_size=RECON_PATCH_SIZE)
        w_raw_noisy_patch_rec = _patchify_1d(w_raw_noisy, patch_size=RECON_PATCH_SIZE)
        w_raw_clean_patch_rec = _patchify_1d(w_raw_clean, patch_size=RECON_PATCH_SIZE)
        anom_mask_patch_rec = _patchify_1d(anom_mask.astype(np.float32), patch_size=RECON_PATCH_SIZE)

        tf_patch_rec = _patchify_2d(tf, patch_size=RECON_PATCH_SIZE)
        time_mean_rec = np.mean(tf_patch_rec, axis=1).astype(np.float32)

        U_30m = np.sum(_patchify_1d(u, patch_size=RECON_PATCH_SIZE), axis=1, keepdims=True).astype(np.float32)
        U_30m_g = (U_30m - self.global_stats.rain_hour_logsum_mean * 0.5) / (
            self.global_stats.rain_hour_logsum_std * 0.5 + EPS
        )

        cov_patch_rec = np.concatenate([U_30m_g, time_mean_rec], axis=-1).astype(np.float32)

        if self.deterministic_rec_hour:
            rec_mask_idx = self._rec_hour_cursor
            self._rec_hour_cursor = (self._rec_hour_cursor + 1) % N_RECON_HIST_PATCH
        else:
            rec_mask_idx = int(self.rng.integers(0, N_RECON_HIST_PATCH))

        rec_query_pos = rec_mask_idx
        rec_query_cov = cov_patch_rec[rec_mask_idx]
        rec_target_patch = z_clean_patch_rec[rec_mask_idx]
        rec_target_raw_patch = w_raw_clean_patch_rec[rec_mask_idx]
        rec_query_raw_patch = w_raw_noisy_patch_rec[rec_mask_idx]
        rec_query_anom_ratio = float(np.mean(anom_mask_patch_rec[rec_mask_idx]))

        visible_idx = [p for p in range(N_RECON_PATCH) if p != rec_mask_idx]
        rec_visible_z_patches = z_noisy_patch_rec[visible_idx]
        rec_visible_cov_patches = cov_patch_rec[visible_idx]
        rec_visible_raw_patches = w_raw_noisy_patch_rec[visible_idx]

        rec_all_hist_anom_ratios = anom_mask_patch_rec[:N_RECON_HIST_PATCH].mean(axis=1).astype(np.float32)

        batch: Dict[str, Any] = {
            "station_id": station_id,
            "t0": torch.tensor(t0, dtype=torch.long),

            "stats_vec": torch.from_numpy(stats_vec),
            "cov_patches": torch.from_numpy(cov_patch_pred),

            "hist_z_patches": torch.from_numpy(hist_z_patches),
            "fut_z_patches": torch.from_numpy(fut_z_patches),

            "hist_raw_patches": torch.from_numpy(_patchify_1d(w_raw_noisy[:HIST_STEPS], PRED_PATCH_SIZE)),
            "fut_raw_patches": torch.from_numpy(_patchify_1d(w_raw_clean[HIST_STEPS:], PRED_PATCH_SIZE)),

            "rec_mask_patch_idx": torch.tensor(rec_mask_idx, dtype=torch.long),
            "rec_query_pos": torch.tensor(rec_query_pos, dtype=torch.long),
            "rec_query_cov": torch.from_numpy(rec_query_cov),
            "rec_target_patch": torch.from_numpy(rec_target_patch),
            "rec_target_raw_patch": torch.from_numpy(rec_target_raw_patch),
            "rec_query_raw_patch": torch.from_numpy(rec_query_raw_patch),
            "rec_visible_z_patches": torch.from_numpy(rec_visible_z_patches),
            "rec_visible_cov_patches": torch.from_numpy(rec_visible_cov_patches),
            "rec_visible_raw_patches": torch.from_numpy(rec_visible_raw_patches),

            "rec_patch_size": torch.tensor(RECON_PATCH_SIZE, dtype=torch.long),
            "rec_anom_point_mask": torch.from_numpy(anom_mask.astype(np.float32)),
            "rec_anom_point_ratio": torch.tensor(float(np.mean(anom_mask)), dtype=torch.float32),
            "rec_query_anom_ratio": torch.tensor(rec_query_anom_ratio, dtype=torch.float32),
            "rec_is_identity": torch.tensor(1 if rec_is_identity else 0, dtype=torch.long),
            "rec_n_events": torch.tensor(rec_n_events, dtype=torch.long),


            "rec_all_hist_z_clean": torch.from_numpy(z_clean_patch_rec[:N_RECON_HIST_PATCH]),
            "rec_all_hist_z_noisy": torch.from_numpy(z_noisy_patch_rec[:N_RECON_HIST_PATCH]),
            "rec_all_hist_raw_clean": torch.from_numpy(w_raw_clean_patch_rec[:N_RECON_HIST_PATCH]),
            "rec_all_hist_raw_noisy": torch.from_numpy(w_raw_noisy_patch_rec[:N_RECON_HIST_PATCH]),
            "rec_all_hist_cov": torch.from_numpy(cov_patch_rec[:N_RECON_HIST_PATCH]),
            "rec_all_hist_anom_ratios": torch.from_numpy(rec_all_hist_anom_ratios),

            "rec_fut_z_noisy": torch.from_numpy(z_noisy_patch_rec[N_RECON_HIST_PATCH:]),
            "rec_fut_raw_noisy": torch.from_numpy(w_raw_noisy_patch_rec[N_RECON_HIST_PATCH:]),
            "rec_fut_cov": torch.from_numpy(cov_patch_rec[N_RECON_HIST_PATCH:]),
        }

        return batch


def collate_two_pass(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["station_id"] = [b.get("station_id") for b in batch_list]

    keys = [k for k in batch_list[0].keys() if k != "station_id"]
    for k in keys:
        v0 = batch_list[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
        else:
            out[k] = [b[k] for b in batch_list]
    return out
