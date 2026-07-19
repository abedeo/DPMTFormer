from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_injector import AnomalyInjectConfig, AnomalyInjector, OscillationConfig, RampConfig, SpikeConfig
from loader import LABEL_COL, WATER_COL


def _valid_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: int | None = None
    for i, valid in enumerate(mask.astype(bool, copy=False)):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _build_cfg(args: argparse.Namespace) -> AnomalyInjectConfig:
    return AnomalyInjectConfig(
        enabled=True,
        identity_prob=args.identity_prob,
        point_ratio_min=args.anom_point_ratio_min,
        point_ratio_max=args.anom_point_ratio_max,
        local_scale_window=args.anom_local_scale_window,
        min_local_scale=args.anom_min_local_scale,
        clip_min=args.anom_clip_min,
        clip_max=args.anom_clip_max,
        max_events_per_window=args.anom_max_events,
        max_tries_per_event=args.anom_max_tries_per_event,
        spike=SpikeConfig(
            prob=args.spike_prob,
            count_min=args.spike_count_min,
            count_max=args.spike_count_max,
            width_min=args.spike_width_min,
            width_max=args.spike_width_max,
            amp_k_min=args.spike_amp_k_min,
            amp_k_max=args.spike_amp_k_max,
        ),
        oscillation=OscillationConfig(
            prob=args.osc_prob,
            dur_min=args.osc_dur_min,
            dur_max=args.osc_dur_max,
            amp_k_min=args.osc_amp_k_min,
            amp_k_max=args.osc_amp_k_max,
            period_min=args.osc_period_min,
            period_max=args.osc_period_max,
            noise_ratio=args.osc_noise_ratio,
        ),
        ramp=RampConfig(
            prob=args.ramp_prob,
            dur_min=args.ramp_dur_min,
            dur_max=args.ramp_dur_max,
            amp_k_min=args.ramp_amp_k_min,
            amp_k_max=args.ramp_amp_k_max,
            shape_prob_step=args.ramp_step_prob,
        ),
    )


def _inject_frame(
    df: pd.DataFrame,
    cfg: AnomalyInjectConfig,
    seed: int,
    window_len: int,
    stride: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if WATER_COL not in df.columns:
        raise ValueError(f"missing required column: {WATER_COL}")

    water = pd.to_numeric(df[WATER_COL], errors="coerce").to_numpy(dtype=np.float32)
    noisy = water.copy()
    label = np.zeros(water.shape[0], dtype=np.int8)
    injector = AnomalyInjector(config=cfg, seed=seed)
    window_len = max(2, int(window_len))
    stride = max(1, int(stride))
    total_windows = 0
    non_identity_windows = 0

    for seg_start, seg_end in _valid_segments(np.isfinite(water)):
        if seg_end - seg_start < window_len:
            continue
        starts = list(range(seg_start, seg_end - window_len + 1, stride))
        last_start = seg_end - window_len
        if starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            end = start + window_len
            injected, mask, meta = injector.inject(water[start:end])
            # Add each event delta so overlapping windows retain every injected event.
            noisy[start:end] += injected - water[start:end]
            label[start:end] = np.maximum(label[start:end], mask.astype(np.int8))
            total_windows += 1
            non_identity_windows += int(not bool(meta.get("is_identity", False)))

    out = df.copy()
    out[WATER_COL] = noisy
    out[LABEL_COL] = label
    valid_points = int(np.isfinite(water).sum())
    summary = {
        "rows": int(len(out)),
        "valid_points": valid_points,
        "anomaly_points": int(label.sum()),
        "anomaly_ratio_valid": float(label.sum() / max(valid_points, 1)),
        "windows": total_windows,
        "non_identity_windows": non_identity_windows,
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy a clean test split and inject deterministic anomalies into it.")
    ap.add_argument("--src_test_dir", type=str, default="data/test")
    ap.add_argument("--out_test_dir", type=str, default="data_anom/test")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--window_len", type=int, default=180)
    ap.add_argument("--stride", type=int, default=180)

    ap.add_argument("--identity_prob", type=float, default=0.40)
    ap.add_argument("--anom_point_ratio_min", type=float, default=0.02)
    ap.add_argument("--anom_point_ratio_max", type=float, default=0.05)
    ap.add_argument("--anom_local_scale_window", type=int, default=36)
    ap.add_argument("--anom_min_local_scale", type=float, default=0.02)
    ap.add_argument("--anom_clip_min", type=float, default=None)
    ap.add_argument("--anom_clip_max", type=float, default=None)
    ap.add_argument("--anom_max_events", type=int, default=3)
    ap.add_argument("--anom_max_tries_per_event", type=int, default=30)
    ap.add_argument("--spike_prob", type=float, default=0.35)
    ap.add_argument("--spike_count_min", type=int, default=1)
    ap.add_argument("--spike_count_max", type=int, default=4)
    ap.add_argument("--spike_width_min", type=int, default=1)
    ap.add_argument("--spike_width_max", type=int, default=2)
    ap.add_argument("--spike_amp_k_min", type=float, default=1.2)
    ap.add_argument("--spike_amp_k_max", type=float, default=2.5)
    ap.add_argument("--osc_prob", type=float, default=0.35)
    ap.add_argument("--osc_dur_min", type=int, default=6)
    ap.add_argument("--osc_dur_max", type=int, default=12)
    ap.add_argument("--osc_amp_k_min", type=float, default=1.2)
    ap.add_argument("--osc_amp_k_max", type=float, default=2.5)
    ap.add_argument("--osc_period_min", type=int, default=2)
    ap.add_argument("--osc_period_max", type=int, default=6)
    ap.add_argument("--osc_noise_ratio", type=float, default=0.12)
    ap.add_argument("--ramp_prob", type=float, default=0.30)
    ap.add_argument("--ramp_dur_min", type=int, default=8)
    ap.add_argument("--ramp_dur_max", type=int, default=12)
    ap.add_argument("--ramp_amp_k_min", type=float, default=1.2)
    ap.add_argument("--ramp_amp_k_max", type=float, default=2.5)
    ap.add_argument("--ramp_step_prob", type=float, default=0.4)
    args = ap.parse_args()

    src_test_root = os.path.abspath(args.src_test_dir)
    out_test_root = os.path.abspath(args.out_test_dir)
    if not os.path.isdir(src_test_root):
        raise FileNotFoundError(f"source test directory does not exist: {src_test_root}")
    if src_test_root == out_test_root:
        raise ValueError("source and output test directories must differ")
    if os.path.exists(out_test_root):
        if not args.overwrite:
            raise FileExistsError(f"output exists, pass --overwrite to replace: {out_test_root}")
        shutil.rmtree(out_test_root)

    shutil.copytree(src_test_root, out_test_root)
    cfg = _build_cfg(args)
    summaries: Dict[str, Dict[str, Any]] = {}
    stations = sorted(name for name in os.listdir(out_test_root) if os.path.isdir(os.path.join(out_test_root, name)))
    for station_idx, station in enumerate(stations):
        path = os.path.join(out_test_root, station, "clean.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        out_df, summary = _inject_frame(
            df=df,
            cfg=cfg,
            seed=int(args.seed) + station_idx * 1009,
            window_len=args.window_len,
            stride=args.stride,
        )
        out_df.to_csv(path, index=False)
        summaries[station] = summary
        print(
            f"{station}: windows={summary['windows']} anomaly_points={summary['anomaly_points']} "
            f"ratio={summary['anomaly_ratio_valid']:.4f}"
        )

    manifest = {
        "src_test_dir": src_test_root,
        "out_test_dir": out_test_root,
        "seed": int(args.seed),
        "window_len": int(args.window_len),
        "stride": int(args.stride),
        "config": asdict(cfg),
        "test_stations": summaries,
    }
    with open(os.path.join(out_test_root, "test_anomaly_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {out_test_root}")


if __name__ == "__main__":
    main()
