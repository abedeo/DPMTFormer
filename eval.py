from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomaly_injector import AnomalyInjectConfig, OscillationConfig, RampConfig, SpikeConfig
from dataset import N_RECON_HIST_PATCH, X3TwoPassDataset, collate_two_pass, compute_global_norm_stats
from loader import load_datasets
from loss import det_hard_label, recon_loss, student_t_nll
from model import DPMTFormer, DPMTFormerConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "data"))


class _StreamingQuantileSummary:
    def __init__(self, sample_cap: int = 200000, seed: int = 2026) -> None:
        self.sample_cap = int(sample_cap)
        self.count = 0
        self.min_value = float("nan")
        self.max_value = float("nan")
        self._rng = np.random.default_rng(seed)
        self._sample = np.empty((self.sample_cap,), dtype=np.float32)
        self._filled = 0

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        values = tensor.detach().float().cpu().reshape(-1).numpy()
        if values.size == 0:
            return

        cur_max = float(np.max(values))
        cur_min = float(np.min(values))
        if np.isnan(self.min_value) or cur_min < self.min_value:
            self.min_value = cur_min
        if np.isnan(self.max_value) or cur_max > self.max_value:
            self.max_value = cur_max

        for value in values:
            if self._filled < self.sample_cap:
                self._sample[self._filled] = value
                self._filled += 1
            else:
                j = int(self._rng.integers(0, self.count + 1))
                if j < self.sample_cap:
                    self._sample[j] = value
            self.count += 1

    def quantile(self, p: float) -> float:
        if self._filled == 0:
            return float("nan")
        return float(np.quantile(self._sample[: self._filled], p))

    def max(self) -> float:
        return self.max_value

    def min(self) -> float:
        return self.min_value


def _autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_anomaly_cfg(args: argparse.Namespace) -> AnomalyInjectConfig:
    return AnomalyInjectConfig(
        enabled=bool(args.eval_with_synth_anomaly),
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


def _load_light_checkpoint(model: torch.nn.Module, ckpt_dir_or_file: str, device: torch.device) -> int:
    if os.path.isdir(ckpt_dir_or_file):
        model_full_path = os.path.join(ckpt_dir_or_file, "model_full.pt")
        meta_path = os.path.join(ckpt_dir_or_file, "meta.json")
        step = 0

        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                step = int(meta.get("step", 0))

        if os.path.isfile(model_full_path):
            state_dict = torch.load(model_full_path, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"[warn] unexpected keys: {unexpected[:8]}")
            if missing:
                print(f"[info] missing keys count: {len(missing)}")
            return step

        raise FileNotFoundError(f"checkpoint missing model_full.pt: {model_full_path}")

    payload = torch.load(ckpt_dir_or_file, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=False)
        return int(payload.get("step", 0))

    model.load_state_dict(payload, strict=False)
    return 0


def _binary_prf1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _scan_best_threshold(scores: np.ndarray, labels: np.ndarray, n_steps: int = 121) -> Dict[str, float]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask].astype(np.float64)
    labels = labels[mask].astype(np.int32)

    if scores.size == 0:
        return {
            "cls_best_threshold": float("nan"),
            "cls_best_precision": float("nan"),
            "cls_best_recall": float("nan"),
            "cls_best_f1": float("nan"),
        }

    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    ths = np.array([score_min]) if score_max <= score_min else np.linspace(score_min, score_max, n_steps)

    best = {"th": float("nan"), "p": 0.0, "r": 0.0, "f1": -1.0}
    for th in ths:
        pred = (scores >= th).astype(np.int32)
        p, r, f1 = _binary_prf1(labels, pred)
        if f1 > best["f1"]:
            best = {"th": float(th), "p": float(p), "r": float(r), "f1": float(f1)}

    return {
        "cls_best_threshold": best["th"],
        "cls_best_precision": best["p"],
        "cls_best_recall": best["r"],
        "cls_best_f1": best["f1"],
    }


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask].astype(np.float64)
    labels = labels[mask].astype(np.int32)
    n_pos = int(labels.sum())
    if scores.size == 0 or n_pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        if r > prev_recall:
            ap += float(p) * float(r - prev_recall)
            prev_recall = float(r)
    return float(ap)


def _metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask].astype(np.float64)
    labels = labels[mask].astype(np.int32)
    if scores.size == 0:
        return {
            "cls_fixed_threshold": float(threshold),
            "cls_fixed_precision": float("nan"),
            "cls_fixed_recall": float("nan"),
            "cls_fixed_f1": float("nan"),
        }
    pred = (scores >= threshold).astype(np.int32)
    p, r, f1 = _binary_prf1(labels, pred)
    return {
        "cls_fixed_threshold": float(threshold),
        "cls_fixed_precision": float(p),
        "cls_fixed_recall": float(r),
        "cls_fixed_f1": float(f1),
    }


def _expand_batch_selected_recon(
    batch: Dict[str, torch.Tensor],
    qidx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    bsz = batch["stats_vec"].shape[0]
    hist_n = batch["rec_all_hist_z_clean"].shape[1]
    device = batch["stats_vec"].device
    qidx = qidx.to(device=device, dtype=torch.long)
    if qidx.ndim == 1:
        qidx = qidx.unsqueeze(0).expand(bsz, -1)
    recon_k = qidx.shape[1]

    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        if key in {
            "rec_all_hist_z_clean",
            "rec_all_hist_z_noisy",
            "rec_all_hist_raw_clean",
            "rec_all_hist_raw_noisy",
            "rec_all_hist_cov",
            "rec_all_hist_anom_ratios",
            "rec_fut_z_noisy",
            "rec_fut_raw_noisy",
            "rec_fut_cov",
        }:
            continue
        out[key] = value.repeat_interleave(recon_k, dim=0)

    hz_clean = batch["rec_all_hist_z_clean"]
    hz_noisy = batch["rec_all_hist_z_noisy"]
    hraw_clean = batch["rec_all_hist_raw_clean"]
    hraw_noisy = batch["rec_all_hist_raw_noisy"]
    hcov = batch["rec_all_hist_cov"]
    hanom = batch["rec_all_hist_anom_ratios"]

    fz_noisy = batch["rec_fut_z_noisy"]
    fraw_noisy = batch["rec_fut_raw_noisy"]
    fcov = batch["rec_fut_cov"]

    patch_dim = hz_clean.shape[-1]
    cov_dim = hcov.shape[-1]
    fut_n = fz_noisy.shape[1]

    b_idx = torch.arange(bsz, device=device).unsqueeze(1).repeat(1, recon_k).reshape(-1)
    m_idx = qidx.reshape(-1)

    out["rec_mask_patch_idx"] = m_idx
    out["rec_query_pos"] = m_idx
    out["rec_query_cov"] = hcov[b_idx, m_idx, :]
    out["rec_target_patch"] = hz_clean[b_idx, m_idx, :]
    out["rec_target_raw_patch"] = hraw_clean[b_idx, m_idx, :]
    out["rec_query_raw_patch"] = hraw_noisy[b_idx, m_idx, :]
    out["rec_query_anom_ratio"] = hanom[b_idx, m_idx]

    hist_idx = torch.arange(hist_n, device=device)
    keep_mask = hist_idx.unsqueeze(0) != m_idx.unsqueeze(1)
    keep_idx = hist_idx.unsqueeze(0).expand(m_idx.shape[0], -1)[keep_mask].view(-1, hist_n - 1)

    hz_noisy_exp = hz_noisy[b_idx]
    hraw_noisy_exp = hraw_noisy[b_idx]
    hcov_exp = hcov[b_idx]

    gather_idx_patch = keep_idx.unsqueeze(-1).expand(-1, -1, patch_dim)
    gather_idx_cov = keep_idx.unsqueeze(-1).expand(-1, -1, cov_dim)

    hist_visible_z = torch.gather(hz_noisy_exp, dim=1, index=gather_idx_patch)
    hist_visible_raw = torch.gather(hraw_noisy_exp, dim=1, index=gather_idx_patch)
    hist_visible_cov = torch.gather(hcov_exp, dim=1, index=gather_idx_cov)

    visible_z = torch.cat([hist_visible_z, fz_noisy[b_idx]], dim=1)
    visible_raw = torch.cat([hist_visible_raw, fraw_noisy[b_idx]], dim=1)
    visible_cov = torch.cat([hist_visible_cov, fcov[b_idx]], dim=1)

    out["rec_visible_z_patches"] = visible_z
    out["rec_visible_raw_patches"] = visible_raw
    out["rec_visible_cov_patches"] = visible_cov
    out["rec_group_size"] = recon_k
    out["rec_base_bsz"] = bsz
    return out


def _select_eval_query_idx(
    hist_n: int,
    recon_eval_patches: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    if recon_eval_patches is None or int(recon_eval_patches) >= hist_n:
        return torch.arange(hist_n, device=device, dtype=torch.long)

    k = max(1, int(recon_eval_patches))
    if k == 1:
        return torch.tensor([hist_n // 2], device=device, dtype=torch.long)

    idx = torch.round(torch.linspace(0, hist_n - 1, steps=k, device=device)).to(torch.long)
    idx = torch.unique_consecutive(idx)
    if idx.numel() < k:
        all_idx = torch.arange(hist_n, device=device, dtype=torch.long)
        mask = torch.ones(hist_n, device=device, dtype=torch.bool)
        mask[idx] = False
        idx = torch.cat([idx, all_idx[mask][: k - idx.numel()]], dim=0)
    return idx[:k]


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    batches: int = 200,
    recon_kind: str = "huber",
    huber_delta: float = 1.0,
    raw_recon_kind: str = "huber",
    raw_huber_delta: float = 0.5,
    cls_label_threshold: float = 0.25,
    recon_eval_patches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = _autocast_dtype(device)

    n = 0
    pred_nll_sum = 0.0
    rec_sum = 0.0
    rec_z_sum = 0.0
    rec_raw_sum = 0.0

    abs_err_summary = _StreamingQuantileSummary(seed=2027)
    norm_err_summary = _StreamingQuantileSummary(seed=2028)
    sigma_summary = _StreamingQuantileSummary(seed=2029)
    pred_nll_summary = _StreamingQuantileSummary(seed=2030)
    rec_summary = _StreamingQuantileSummary(seed=2031)
    rec_z_summary = _StreamingQuantileSummary(seed=2032)
    rec_raw_summary = _StreamingQuantileSummary(seed=2033)
    cls_score_all = []
    cls_label_all = []
    selected_qidx = None

    for batch in dl:
        batch = {key: value for key, value in batch.items() if key != "station_id"}
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
        if selected_qidx is None:
            selected_qidx = _select_eval_query_idx(
                hist_n=int(batch["rec_all_hist_z_clean"].shape[1]),
                recon_eval_patches=recon_eval_patches,
                device=device,
            )
        batch_k = _expand_batch_selected_recon(batch, selected_qidx)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out_pred = model.forward_pred(batch)
            out_rec = model.forward_recon(batch_k)
            out_cls = model.forward_cls(batch)
            y = out_pred["target_fut_z"]
            mu = out_pred["pred_params"]["mu"]
            sigma = out_pred["pred_params"]["sigma"]
            nu = out_pred["pred_params"]["nu"]
            rec_hat = out_rec["recon_patch_hat"]
            rec_tgt = out_rec["target_rec_patch"]
            rec_raw_hat = out_rec.get("recon_raw_patch_hat")
            rec_raw_tgt = out_rec.get("target_rec_raw_patch")

        y = y.float()
        mu = mu.float()
        sigma = sigma.float()
        nu = nu.float()
        rec_hat = rec_hat.float()
        rec_tgt = rec_tgt.float()

        nll_elem = student_t_nll(y=y, mu=mu, sigma=sigma, nu=nu, reduce="none")
        nll_mean = nll_elem.mean()

        rec_z_elem = recon_loss(rec_hat, rec_tgt, kind=recon_kind, huber_delta=huber_delta, reduce="none")
        rec_z_mean = rec_z_elem.mean()

        if rec_raw_hat is not None and rec_raw_tgt is not None:
            rec_raw_hat = rec_raw_hat.float()
            rec_raw_tgt = rec_raw_tgt.float()
            rec_raw_elem = recon_loss(
                rec_raw_hat,
                rec_raw_tgt,
                kind=raw_recon_kind,
                huber_delta=raw_huber_delta,
                reduce="none",
            )
            rec_raw_mean = rec_raw_elem.mean()
        else:
            rec_raw_elem = None
            rec_raw_mean = torch.zeros((), dtype=torch.float32, device=device)

        rec_total_mean = rec_z_mean + rec_raw_mean
        pred_nll_sum += float(nll_mean.item())
        rec_sum += float(rec_total_mean.item())
        rec_z_sum += float(rec_z_mean.item())
        rec_raw_sum += float(rec_raw_mean.item())

        abs_err = (y - mu).abs().detach().cpu().reshape(-1)
        norm_err = ((y - mu).abs() / (sigma + 1e-6)).detach().cpu().reshape(-1)
        sigma_cpu = sigma.detach().cpu().reshape(-1)
        nll_cpu = nll_elem.detach().cpu().reshape(-1)
        rec_z_cpu = rec_z_elem.detach().cpu().reshape(-1)

        abs_err_summary.update(abs_err)
        norm_err_summary.update(norm_err)
        sigma_summary.update(sigma_cpu)
        pred_nll_summary.update(nll_cpu)
        rec_z_summary.update(rec_z_cpu)

        if rec_raw_elem is not None:
            rec_raw_cpu = rec_raw_elem.detach().cpu().reshape(-1)
            rec_raw_summary.update(rec_raw_cpu)
            rec_summary.update(torch.cat([rec_z_cpu, rec_raw_cpu], dim=0))
        else:
            rec_summary.update(rec_z_cpu)

        cls_score = out_cls["cls_logits"].float().detach().cpu().reshape(-1).numpy().astype(np.float32)
        cls_label = det_hard_label(
            out_cls["cls_target_anom_ratio"].detach().cpu().reshape(-1),
            pos_threshold=float(cls_label_threshold),
        ).numpy()

        cls_score_all.append(cls_score)
        cls_label_all.append(cls_label)

        n += 1
        if n >= batches:
            break

    metrics = {
        "eval_batches": float(n),
        "recon_eval_patches": float(selected_qidx.numel() if selected_qidx is not None else N_RECON_HIST_PATCH),
        "pred_nll_mean": pred_nll_sum / max(n, 1),
        "pred_nll_p50": pred_nll_summary.quantile(0.50),
        "pred_nll_p90": pred_nll_summary.quantile(0.90),
        "pred_nll_p99": pred_nll_summary.quantile(0.99),
        "abs_err_p50": abs_err_summary.quantile(0.50),
        "abs_err_p90": abs_err_summary.quantile(0.90),
        "abs_err_p99": abs_err_summary.quantile(0.99),
        "abs_err_max": abs_err_summary.max(),
        "norm_err_p50": norm_err_summary.quantile(0.50),
        "norm_err_p90": norm_err_summary.quantile(0.90),
        "norm_err_p99": norm_err_summary.quantile(0.99),
        "norm_err_max": norm_err_summary.max(),
        "sigma_p10": sigma_summary.quantile(0.10),
        "sigma_p50": sigma_summary.quantile(0.50),
        "sigma_p90": sigma_summary.quantile(0.90),
        "sigma_min": sigma_summary.min(),
        "sigma_max": sigma_summary.max(),
        "rec_mean": rec_sum / max(n, 1),
        "rec_p50": rec_summary.quantile(0.50),
        "rec_p90": rec_summary.quantile(0.90),
        "rec_p99": rec_summary.quantile(0.99),
        "rec_z_mean": rec_z_sum / max(n, 1),
        "rec_z_p50": rec_z_summary.quantile(0.50),
        "rec_z_p90": rec_z_summary.quantile(0.90),
        "rec_z_p99": rec_z_summary.quantile(0.99),
        "rec_raw_mean": rec_raw_sum / max(n, 1),
        "rec_raw_p50": rec_raw_summary.quantile(0.50),
        "rec_raw_p90": rec_raw_summary.quantile(0.90),
        "rec_raw_p99": rec_raw_summary.quantile(0.99),
    }

    if cls_score_all and cls_label_all:
        cls_scores = np.concatenate(cls_score_all, axis=0)
        cls_labels = np.concatenate(cls_label_all, axis=0)
        metrics["cls_pr_auc"] = _average_precision(cls_scores, cls_labels)
        metrics.update(_metrics_at_threshold(cls_scores, cls_labels, threshold=0.0))
        n_pos = int(cls_labels.sum())
        n_total = int(cls_labels.shape[0])
        if n_total > 0 and 0 < n_pos < n_total:
            metrics.update(_scan_best_threshold(cls_scores, cls_labels, n_steps=121))
        else:
            metrics.update(
                {
                    "cls_pr_auc": metrics.get("cls_pr_auc", float("nan")),
                    "cls_best_threshold": float("nan"),
                    "cls_best_precision": float("nan"),
                    "cls_best_recall": float("nan"),
                    "cls_best_f1": float("nan"),
                }
            )
    else:
        metrics.update(
            {
                "cls_pr_auc": float("nan"),
                "cls_fixed_threshold": 0.0,
                "cls_fixed_precision": float("nan"),
                "cls_fixed_recall": float("nan"),
                "cls_fixed_f1": float("nan"),
                "cls_best_threshold": float("nan"),
                "cls_best_precision": float("nan"),
                "cls_best_recall": float("nan"),
                "cls_best_f1": float("nan"),
            }
        )

    model.train()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--split", type=str, default="eval", choices=["train", "eval"])
    ap.add_argument("--ckpt", type=str,default="")
    ap.add_argument("--eval_seed", type=int, default=2026)

    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--sigma_floor", type=float, default=0.05)
    ap.add_argument("--cls_label_threshold", type=float, default=0.15)
    ap.add_argument("--recon_eval_patches", type=int, default=24)

    ap.add_argument("--recon_kind", type=str, default="huber", choices=["huber", "l1", "mse"])
    ap.add_argument("--huber_delta", type=float, default=0.8)
    ap.add_argument("--raw_recon_kind", type=str, default="huber", choices=["huber", "l1", "mse"])
    ap.add_argument("--raw_huber_delta", type=float, default=0.4)

    ap.add_argument("--pred_patch_size", type=int, default=12)
    ap.add_argument("--recon_patch_size", type=int, default=6)
    ap.add_argument("--use_raw_branch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_use_raw_branch", action="store_false", dest="use_raw_branch", help=argparse.SUPPRESS)
    ap.add_argument("--raw_scale", type=float, default=1.0)
    ap.add_argument("--raw_clip", type=float, default=None)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cov_dim", type=int, default=5)

    ap.add_argument("--eval_with_synth_anomaly", action=argparse.BooleanOptionalAction, default=True)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_raw_branch = bool(args.use_raw_branch)

    datasets = load_datasets(args.data_dir)
    gstats = compute_global_norm_stats(datasets["train"])
    anomaly_cfg = _build_anomaly_cfg(args)

    ds = X3TwoPassDataset(
        split_data=datasets[args.split],
        global_stats=gstats,
        samples_per_epoch=args.batch_size * args.batches,
        sigma_floor=args.sigma_floor,
        use_synth_anomaly=bool(args.eval_with_synth_anomaly),
        anomaly_cfg=anomaly_cfg,
        event_prob=0.0,
        seed=args.eval_seed,
    )

    g_dl = torch.Generator()
    g_dl.manual_seed(args.eval_seed)
    if hasattr(ds, "reset_rng"):
        ds.reset_rng(args.eval_seed)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_two_pass,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        generator=g_dl,
        persistent_workers=bool(args.num_workers > 0),
    )

    model = DPMTFormer(
        DPMTFormerConfig(
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

    step = _load_light_checkpoint(model, args.ckpt, device=device)

    metrics = eval_model(
        model=model,
        dl=dl,
        device=device,
        batches=args.batches,
        recon_kind=args.recon_kind,
        huber_delta=args.huber_delta,
        raw_recon_kind=args.raw_recon_kind,
        raw_huber_delta=args.raw_huber_delta,
        cls_label_threshold=args.cls_label_threshold,
        recon_eval_patches=args.recon_eval_patches,
    )
    metrics["ckpt_step"] = float(step)
    metrics["ckpt_path"] = os.path.abspath(args.ckpt)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
