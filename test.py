from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    N_RECON_HIST_PATCH,
    RECON_PATCH_SIZE,
    TOTAL_STEPS,
    X3TwoPassDataset,
    collate_two_pass,
    compute_global_norm_stats,
)
from eval import _autocast_dtype, _binary_prf1, _expand_batch_selected_recon, _load_light_checkpoint
from loader import LABEL_COL, load_split
from model import DPMTFormer, DPMTFormerConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "data"))


class SlidingWindowTestDataset(X3TwoPassDataset):
    def __init__(
        self,
        split_data: Dict[str, dict],
        global_stats: Any,
        stride: int,
        seed: int,
        sigma_floor: float,
    ) -> None:
        self.windows = _build_windows(split_data, stride=stride)
        if not self.windows:
            raise ValueError("no test windows available; check test split segment lengths")

        super().__init__(
            split_data=split_data,
            global_stats=global_stats,
            samples_per_epoch=len(self.windows),
            seed=seed,
            deterministic_rec_hour=True,
            sigma_floor=sigma_floor,
            station_uniform=False,
            seg_weight_by_len=False,
            event_prob=0.0,
            use_synth_anomaly=False,
            use_offline_anomaly_label=True,
        )
        self._fixed_idx = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._fixed_idx = int(idx)
        return super().__getitem__(idx)

    def _sample_window(self):
        station_id, t0 = self.windows[self._fixed_idx]
        df = self.split_data[station_id]["df"]
        return station_id, df, int(t0)


def _build_windows(split_data: Dict[str, dict], stride: int) -> List[Tuple[str, int]]:
    stride = max(1, int(stride))
    windows: List[Tuple[str, int]] = []
    for station_id in sorted(split_data.keys()):
        for seg in split_data[station_id]["segments_kept"]:
            if seg.length < TOTAL_STEPS:
                continue
            max_start = int(seg.end - TOTAL_STEPS + 1)
            starts = list(range(int(seg.start), max_start + 1, stride))
            if starts and starts[-1] != max_start:
                starts.append(max_start)
            for t0 in starts:
                windows.append((station_id, int(t0)))
    return windows


def _prepare_test_split(split_data: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for station_id, item in split_data.items():
        if LABEL_COL not in item["df"].columns:
            raise ValueError(f"{station_id}: missing required offline label column: {LABEL_COL}")
        item_copy = dict(item)
        item_copy["segments_kept"] = [
            seg for seg in item.get("segments_raw", item.get("segments_kept", [])) if seg.length >= TOTAL_STEPS
        ]
        if item_copy["segments_kept"]:
            out[station_id] = item_copy
    return out


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == "station_id":
            out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _add_patch_scores(
    store: Dict[str, Dict[int, Dict[str, Any]]],
    station_ids: List[str],
    t0: torch.Tensor,
    logits: torch.Tensor,
    label_ratio: torch.Tensor,
) -> None:
    logits_np = logits.detach().float().cpu().numpy()
    labels_np = label_ratio.detach().float().cpu().numpy()
    t0_np = t0.detach().cpu().numpy().astype(np.int64)

    bsz, hist_n = logits_np.shape
    for i in range(bsz):
        station = str(station_ids[i])
        station_store = store.setdefault(station, {})
        base_t0 = int(t0_np[i])
        for j in range(hist_n):
            abs_patch_start = base_t0 + int(j) * RECON_PATCH_SIZE
            item = station_store.setdefault(
                abs_patch_start,
                {"logits": [], "label_ratio": 0.0},
            )
            item["logits"].append(float(logits_np[i, j]))
            item["label_ratio"] = max(float(item["label_ratio"]), float(labels_np[i, j]))


def _metrics_from_store(
    station_store: Dict[int, Dict[str, Any]],
    threshold: float,
    cls_label_threshold: float,
) -> Dict[str, float]:
    scores = []
    labels = []
    for item in station_store.values():
        scores.append(float(np.median(item["logits"])))
        labels.append(1 if float(item["label_ratio"]) >= float(cls_label_threshold) else 0)

    if not scores:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_patches": 0.0,
            "n_positive": 0.0,
            "n_pred_positive": 0.0,
        }

    score_arr = np.asarray(scores, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int32)
    pred_arr = (score_arr >= float(threshold)).astype(np.int32)
    precision, recall, f1 = _binary_prf1(label_arr, pred_arr)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_patches": float(label_arr.shape[0]),
        "n_positive": float(label_arr.sum()),
        "n_pred_positive": float(pred_arr.sum()),
    }


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    threshold: float,
    cls_label_threshold: float,
    recon_eval_patches: int,
) -> Dict[str, Any]:
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _autocast_dtype(device)
    selected_qidx = None
    patch_store: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for step, batch in enumerate(dl, start=1):
        station_ids = list(batch["station_id"])
        t0 = batch["t0"].detach().cpu()
        batch = {key: value for key, value in batch.items() if key != "station_id"}
        batch = _move_batch_to_device(batch, device)

        if selected_qidx is None:
            hist_n = int(batch["rec_all_hist_z_clean"].shape[1])
            if recon_eval_patches <= 0 or recon_eval_patches >= hist_n:
                selected_qidx = torch.arange(hist_n, device=device, dtype=torch.long)
            else:
                selected_qidx = torch.round(
                    torch.linspace(0, hist_n - 1, steps=int(recon_eval_patches), device=device)
                ).to(torch.long)

        batch_k = _expand_batch_selected_recon(batch, selected_qidx)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _ = model.forward_pred(batch)
            _ = model.forward_recon(batch_k)
            out_cls = model.forward_cls(batch)

        _add_patch_scores(
            store=patch_store,
            station_ids=station_ids,
            t0=t0,
            logits=out_cls["cls_logits"],
            label_ratio=out_cls["cls_target_anom_ratio"],
        )

        if step % 100 == 0:
            print(f"[test] batches={step} stations={len(patch_store)}")

    per_station = {
        station: _metrics_from_store(items, threshold=threshold, cls_label_threshold=cls_label_threshold)
        for station, items in sorted(patch_store.items())
    }

    merged: Dict[int, Dict[str, Any]] = {}
    offset = 0
    for station, items in sorted(patch_store.items()):
        for patch_start, item in items.items():
            merged[offset + int(patch_start)] = item
        offset += 10_000_000_000

    overall = _metrics_from_store(merged, threshold=threshold, cls_label_threshold=cls_label_threshold)
    return {
        "threshold": float(threshold),
        "cls_label_threshold": float(cls_label_threshold),
        "recon_eval_patches": float(selected_qidx.numel() if selected_qidx is not None else 0),
        "overall": overall,
        "per_station": per_station,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--test_data_dir", type=str, default="./data_anom/test")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=1.92604)
    ap.add_argument("--cls_label_threshold", type=float, default=0.15)
    ap.add_argument("--recon_eval_patches", type=int, default=N_RECON_HIST_PATCH)
    ap.add_argument("--sigma_floor", type=float, default=0.05)

    ap.add_argument("--pred_patch_size", type=int, default=12)
    ap.add_argument("--recon_patch_size", type=int, default=6)
    ap.add_argument("--use_raw_branch", action="store_true", default=True)
    ap.add_argument("--no_use_raw_branch", action="store_true")
    ap.add_argument("--raw_scale", type=float, default=1.0)
    ap.add_argument("--raw_clip", type=float, default=None)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cov_dim", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_raw_branch = not args.no_use_raw_branch

    train_data = load_split(os.path.join(args.data_dir, "train"), "train", scoring=False)
    test_root = args.test_data_dir if args.test_data_dir else os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"missing test split directory: {test_root}")
    test_data = _prepare_test_split(load_split(test_root, "test", scoring=False))
    gstats = compute_global_norm_stats(train_data)

    ds = SlidingWindowTestDataset(
        split_data=test_data,
        global_stats=gstats,
        stride=args.stride,
        seed=args.seed,
        sigma_floor=args.sigma_floor,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_two_pass,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
    )

    model = DPMTFormer(
        DPMTFormerConfig(
            backbone_path="dpmtformer",
            use_text_prompt=False,
            sigma_min=1e-3,
            fixed_nu=8.0,
            n_hist_patch=12,
            n_fut_patch=3,
            pred_patch_size=args.pred_patch_size,
            recon_patch_size=args.recon_patch_size,
            max_cls_pos=30,
            max_patch_pos=15,
            max_recon_pos=30,
            use_raw_branch=use_raw_branch,
            raw_patch_dim=args.pred_patch_size,
            raw_recon_patch_dim=args.recon_patch_size,
            raw_scale=args.raw_scale,
            raw_clip=args.raw_clip,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            cov_dim=args.cov_dim,
        )
    )
    model.to(device)
    ckpt_step = _load_light_checkpoint(model, args.ckpt, device=device)

    metrics = run_test(
        model=model,
        dl=dl,
        device=device,
        threshold=args.threshold,
        cls_label_threshold=args.cls_label_threshold,
        recon_eval_patches=args.recon_eval_patches,
    )
    metrics["ckpt_step"] = float(ckpt_step)
    metrics["ckpt_path"] = os.path.abspath(args.ckpt)
    metrics["test_windows"] = float(len(ds))
    metrics["stride"] = float(args.stride)

    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(payload)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
