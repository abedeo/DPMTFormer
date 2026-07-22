import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


RAIN_COL = "rain_diff"
WATER_COL = "current_water"
DATE_COL = "date"
TIME_COL = "time"
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


def load_one_x3(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}，实际列: {list(df.columns)}")

    out = df[REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]].copy()
    for c in REQUIRED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_split(root: str, split_name: str) -> Dict[str, dict]:
    stations = find_station_files(root)
    if not stations:
        print(f"[WARN] {split_name}: 未找到 clean.csv，期望路径: {root}/<id>/clean.csv")
        return {}

    data: Dict[str, dict] = {}

    for station_id, path in stations:
        df = load_one_x3(path)

        all_segments = split_by_water_nan(df[WATER_COL])
        kept_segments = [s for s in all_segments if s.length >= MIN_LEN]

        data[station_id] = {
            "df": df,
            "segments_raw": all_segments,
            "segments_kept": kept_segments,
        }

    return data


def load_datasets(data_dir: str = "./data") -> Dict[str, Dict[str, dict]]:
    train_root = os.path.join(data_dir, "train")
    eval_root = os.path.join(data_dir, "eval")

    return {
        "train": load_split(train_root, "train"),
        "eval": load_split(eval_root, "eval"),
    }
