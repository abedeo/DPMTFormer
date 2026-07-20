from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAIN_COL = "rain_diff"
WATER_COL = "current_water"
DATE_COL = "date"
TIME_COL = "time"
REQUIRED_COLUMNS = (RAIN_COL, WATER_COL)
STEPS_PER_DAY = 24 * 12


@dataclass(frozen=True)
class CoarseProfile:
    wet_after_dry: float
    wet_after_wet: float
    log_rain_mean: float
    log_rain_std: float
    water_step_scale: float
    water_level_center: float
    water_level_station_scale: float


@dataclass(frozen=True)
class SyntheticStationProfile:
    base_level: float
    annual_amplitude: float
    response_hours: float
    autoregression: float
    response_multiplier: float
    noise_multiplier: float
    rain_multiplier: float


def _iter_source_paths(source_dir: Path) -> Iterable[Path]:
    yield from sorted(source_dir.glob("*/clean.csv"))


def _robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.05
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, 1e-4)


def _quantize(value: float, step: float, low: float, high: float) -> float:
    value = float(np.clip(value, low, high))
    return float(np.clip(round(value / step) * step, low, high))


def estimate_coarse_profile(source_dir: Path | str) -> CoarseProfile:
    source_dir = Path(source_dir)
    paths = list(_iter_source_paths(source_dir))
    if not paths:
        raise FileNotFoundError(f"No <station>/clean.csv files found under {source_dir}")

    dry_to_wet = 0
    dry_count = 0
    wet_to_wet = 0
    wet_count = 0
    log_rain_chunks: list[np.ndarray] = []
    water_diff_chunks: list[np.ndarray] = []
    station_water_centers: list[float] = []

    for path in paths:
        frame = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        rain = pd.to_numeric(frame[RAIN_COL], errors="coerce").to_numpy(dtype=np.float64)
        water = pd.to_numeric(frame[WATER_COL], errors="coerce").to_numpy(dtype=np.float64)

        rain = np.maximum(rain, 0.0)
        valid_rain = np.isfinite(rain)
        wet = (rain > 0.0) & valid_rain
        pairs = valid_rain[:-1] & valid_rain[1:]
        prev_wet = wet[:-1][pairs]
        next_wet = wet[1:][pairs]
        dry_count += int((~prev_wet).sum())
        dry_to_wet += int((~prev_wet & next_wet).sum())
        wet_count += int(prev_wet.sum())
        wet_to_wet += int((prev_wet & next_wet).sum())

        positive_rain = rain[wet]
        if positive_rain.size:
            log_rain_chunks.append(np.log1p(positive_rain[:20_000]))

        finite_water = water[np.isfinite(water)]
        if finite_water.size:
            station_water_centers.append(float(np.median(finite_water)))

        valid_water_pairs = np.isfinite(water[:-1]) & np.isfinite(water[1:])
        if valid_water_pairs.any():
            water_diff_chunks.append(np.diff(water)[valid_water_pairs][:50_000])

    if not log_rain_chunks or not water_diff_chunks or not station_water_centers:
        raise ValueError("Source data do not contain enough finite rainfall and water-level observations")

    log_rain = np.concatenate(log_rain_chunks)
    water_diff = np.concatenate(water_diff_chunks)

    raw_p01 = dry_to_wet / max(dry_count, 1)
    raw_p11 = wet_to_wet / max(wet_count, 1)
    raw_mean = float(np.mean(log_rain))
    raw_std = float(np.std(log_rain))
    raw_step_scale = _robust_scale(water_diff)
    raw_water_center = float(np.median(station_water_centers))
    raw_water_station_scale = _robust_scale(np.asarray(station_water_centers, dtype=np.float64))


    return CoarseProfile(
        wet_after_dry=_quantize(raw_p01, 0.05, 0.02, 0.40),
        wet_after_wet=_quantize(raw_p11, 0.05, 0.20, 0.95),
        log_rain_mean=_quantize(raw_mean, 0.10, 0.10, 3.00),
        log_rain_std=_quantize(raw_std, 0.10, 0.20, 2.00),
        water_step_scale=_quantize(raw_step_scale, 0.01, 0.01, 0.50),
        water_level_center=_quantize(raw_water_center, 0.50, 0.50, 30.00),
        water_level_station_scale=_quantize(raw_water_station_scale, 0.25, 0.25, 6.00),
    )


def _calendar_features(timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    day_of_year = timestamps.dayofyear.to_numpy(dtype=np.float64)
    minutes = (timestamps.hour * 60 + timestamps.minute).to_numpy(dtype=np.float64)
    return day_of_year / 365.0, minutes / (24.0 * 60.0)


def _simulate_rain(
    timestamps: pd.DatetimeIndex,
    profile: CoarseProfile,
    rng: np.random.Generator,
    station: SyntheticStationProfile,
) -> np.ndarray:
    n = len(timestamps)
    annual_phase = 2.0 * np.pi * (timestamps.dayofyear.to_numpy(dtype=np.float64) / 365.0)
    seasonal_wet_multiplier = np.clip(1.0 + 0.35 * np.sin(annual_phase - 0.8), 0.55, 1.45)
    rain = np.zeros(n, dtype=np.float64)
    was_wet = False

    for index in range(n):
        probability = profile.wet_after_wet if was_wet else profile.wet_after_dry
        probability = float(np.clip(probability * seasonal_wet_multiplier[index], 0.01, 0.98))
        was_wet = bool(rng.random() < probability)
        if was_wet:
            log_amount = rng.normal(profile.log_rain_mean, profile.log_rain_std)
            rain[index] = max(0.0, math.expm1(log_amount)) * station.rain_multiplier

    return rain.astype(np.float32)


def _simulate_water(
    timestamps: pd.DatetimeIndex,
    rain: np.ndarray,
    profile: CoarseProfile,
    rng: np.random.Generator,
    station: SyntheticStationProfile,
) -> np.ndarray:
    n = len(timestamps)
    annual_phase = 2.0 * np.pi * (timestamps.dayofyear.to_numpy(dtype=np.float64) / 365.0)
    time_phase = 2.0 * np.pi * (
        (timestamps.hour * 60 + timestamps.minute).to_numpy(dtype=np.float64) / (24.0 * 60.0)
    )

    baseline = (
        station.base_level
        + station.annual_amplitude * np.sin(annual_phase - 1.1)
        + 0.03 * np.sin(time_phase)
    )

    kernel_steps = max(12, int(station.response_hours * 12 * 4))
    kernel = np.exp(-np.arange(kernel_steps, dtype=np.float64) / (station.response_hours * 12.0))
    kernel /= kernel.sum()
    rain_drive = np.convolve(np.log1p(rain.astype(np.float64)), kernel, mode="full")[:n]

    residual = np.zeros(n, dtype=np.float64)
    response_strength = profile.water_step_scale * 30.0 * station.response_multiplier
    noise_scale = profile.water_step_scale * 0.45 * station.noise_multiplier
    for index in range(1, n):
        residual[index] = (
            station.autoregression * residual[index - 1]
            + response_strength * rain_drive[index]
            + rng.normal(0.0, noise_scale)
        )

    water = baseline + residual
    return np.maximum(water, 0.05).astype(np.float32)


def generate_split(
    start: datetime,
    end: datetime,
    profile: CoarseProfile,
    rng: np.random.Generator,
    station: SyntheticStationProfile,
) -> pd.DataFrame:
    warmup_start = start - timedelta(days=21)
    all_timestamps = pd.date_range(warmup_start, end, freq="5min", inclusive="left")
    rain = _simulate_rain(all_timestamps, profile, rng, station)
    water = _simulate_water(all_timestamps, rain, profile, rng, station)

    keep = all_timestamps >= pd.Timestamp(start)
    timestamps = all_timestamps[keep]
    date, time = _calendar_features(timestamps)
    return pd.DataFrame(
        {
            RAIN_COL: rain[keep],
            WATER_COL: water[keep],
            DATE_COL: date.astype(np.float32),
            TIME_COL: time.astype(np.float32),
        }
    )


def _sample_station_profile(rng: np.random.Generator, profile: CoarseProfile) -> SyntheticStationProfile:
    return SyntheticStationProfile(
        base_level=max(
            0.05,
            float(profile.water_level_center + rng.normal(0.0, profile.water_level_station_scale)),
        ),
        annual_amplitude=float(rng.uniform(0.20, 0.70)),
        response_hours=float(rng.uniform(6.0, 16.0)),
        autoregression=float(rng.uniform(0.984, 0.995)),
        response_multiplier=float(rng.uniform(0.80, 1.25)),
        noise_multiplier=float(rng.uniform(0.80, 1.25)),
        rain_multiplier=float(rng.uniform(0.80, 1.20)),
    )


def write_dataset(output_dir: Path, profile: CoarseProfile, station_count: int, seed: int) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"existing output directory: {output_dir}. "
        )

    split_dates = {
        "train": (datetime(2021, 1, 1), datetime(2021, 7, 1)),
        "eval": (datetime(2022, 1, 1), datetime(2022, 2, 1)),
        "test": (datetime(2022, 2, 1), datetime(2022, 4, 1)),
    }
    stations = [
        _sample_station_profile(np.random.default_rng(seed + 1_000_000 + station_index), profile)
        for station_index in range(station_count)
    ]

    for split_index, (split, (start, end)) in enumerate(split_dates.items()):
        for station_index in range(station_count):
            split_seed = seed + 10_000 * split_index + station_index
            frame = generate_split(start, end, profile, np.random.default_rng(split_seed), stations[station_index])
            station_dir = output_dir / split / f"synth_{station_index + 1:02d}"
            station_dir.mkdir(parents=True, exist_ok=False)
            frame.to_csv(station_dir / "clean.csv", index=False, float_format="%.7g")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic water-level demo data.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/cp"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_demo"),
    )
    parser.add_argument("--stations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.stations <= 0:
        parser.error("--stations must be positive")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    profile = estimate_coarse_profile(source_dir)
    write_dataset(output_dir, profile, station_count=args.stations, seed=args.seed)
    print(f"Created synthetic demo dataset at {output_dir} with {args.stations} station(s).")


if __name__ == "__main__":
    main()
