import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


RAIN_COL = "rain_diff"
WATER_COL = "current_water"
DATE_COL = "date"
TIME_COL = "time"
ORIGINAL_WATER_COL = "original_water"
LABEL_COL = "label"

REQUIRED_COLS = [RAIN_COL, WATER_COL, DATE_COL, TIME_COL]
OPTIONAL_COLS = ["dt", LABEL_COL]
MIN_LEN = 360


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    length: int


def find_station_files(root: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(root, "*", "clean.csv")
    paths = sorted(glob.glob(pattern))
    return [(os.path.basename(os.path.dirname(p)), p) for p in paths]


def split_by_water_nan(water: pd.Series) -> List[Segment]:
    is_valid = water.notna().to_numpy(dtype=bool)
    n = len(is_valid)
    segments: List[Segment] = []

    i = 0
    while i < n:
        if not is_valid[i]:
            i += 1
            continue
        j = i
        while j < n and is_valid[j]:
            j += 1
        segments.append(Segment(start=i, end=j - 1, length=j - i))
        i = j

    return segments


def _robust_fill_water_4nbr(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32, copy=True)
    n = w.shape[0]
    isnan = ~np.isfinite(w)

    if not np.any(isnan):
        return w

    valid_idx = np.where(~isnan)[0]
    if valid_idx.size == 0:
        return np.zeros_like(w, dtype=np.float32)

    missing_idx = np.where(isnan)[0]
    for i in missing_idx:
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]

        left_vals = w[left[-2:]] if left.size > 0 else np.array([], dtype=np.float32)
        right_vals = w[right[:2]] if right.size > 0 else np.array([], dtype=np.float32)

        nbrs = np.concatenate([left_vals, right_vals], axis=0)
        nbrs = nbrs[np.isfinite(nbrs)]

        if nbrs.size >= 4:
            s = np.sort(nbrs)
            w[i] = float(0.5 * (s[1] + s[2]))
        elif nbrs.size >= 2:
            s = np.sort(nbrs)
            lo = (s.size - 1) // 2
            hi = s.size // 2
            w[i] = float(0.5 * (s[lo] + s[hi]))
        elif nbrs.size == 1:
            w[i] = float(nbrs[0])
        else:
            w[i] = 0.0

    return w


def load_one_x3(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}，实际列: {list(df.columns)}")

    out = df[REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]].copy()
    for c in REQUIRED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_split(root: str, split_name: str, scoring: bool = False) -> Dict[str, dict]:
    stations = find_station_files(root)
    if not stations:
        print(f"[WARN] {split_name}: 未找到 clean.csv，期望路径: {root}/<id>/clean.csv")
        return {}

    data: Dict[str, dict] = {}

    for station_id, path in stations:
        df = load_one_x3(path)

        w_raw = pd.to_numeric(df[WATER_COL], errors="coerce").to_numpy(dtype=np.float32)
        water_valid_mask = np.isfinite(w_raw)

        if scoring:
            w_fill = _robust_fill_water_4nbr(w_raw)
            df[ORIGINAL_WATER_COL] = w_raw
            df[WATER_COL] = w_fill.astype(np.float32)
        else:
            df[ORIGINAL_WATER_COL] = w_raw

        all_segments = split_by_water_nan(df[WATER_COL])
        kept_segments = [s for s in all_segments if s.length >= MIN_LEN]

        total_rows = len(df)
        usable_points = int(sum(s.length for s in kept_segments))
        usable_ratio = (usable_points / total_rows) if total_rows > 0 else 0.0

        data[station_id] = {
            "path": path,
            "df": df,
            "segments_raw": all_segments,
            "segments_kept": kept_segments,
            "usable_points": usable_points,
            "usable_ratio": usable_ratio,
            "water_valid_mask": water_valid_mask.astype(bool),
        }

    return data


def load_datasets(data_dir: str = "./data", scoring: bool = False) -> Dict[str, Dict[str, dict]]:
    train_root = os.path.join(data_dir, "train")
    eval_root = os.path.join(data_dir, "eval")

    return {
        "train": load_split(train_root, "train", scoring=False),
        "eval": load_split(eval_root, "eval", scoring=False),
    }


if __name__ == "__main__":
    _ = load_datasets("./data")
