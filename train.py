from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from anomaly_injector import AnomalyInjectConfig, OscillationConfig, RampConfig, SpikeConfig
from dataset import N_RECON_HIST_PATCH, X3TwoPassDataset, collate_two_pass, compute_global_norm_stats
from eval import eval_model
from loader import load_datasets
from loss import LossConfig, compute_losses
from model import DPMTFormer, DPMTFormerConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "data"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _count_trainable_params(model: torch.nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


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


@torch.no_grad()
def _sigma_diag_for_console(out: Dict[str, Any]) -> Dict[str, float]:
    sigma = out["pred_params"]["sigma"]
    flat = sigma.detach().float().cpu().reshape(-1)
    return {
        "sig50": float(flat.median()),
        "sig90": float(torch.quantile(flat, 0.90)),
    }


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0

        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def _save_hparams_doc(out_dir: str, args: argparse.Namespace) -> None:
    json_path = os.path.join(out_dir, "hparams.json")
    md_path = os.path.join(out_dir, "hparams.md")
    args_dict = vars(args)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)

    groups = [
        (
            "Experiment",
            [
                "data_dir",
                "train_split",
                "eval_split",
                "out_dir",
                "seed",
                "resume_ckpt",
                "resume_latest",
            ],
        ),
        (
            "Backbone",
            [
                "d_model",
                "n_heads",
                "n_layers",
                "ff_dim",
                "dropout",
                "cov_dim",
                "pred_patch_size",
                "recon_patch_size",
                "use_raw_branch",
                "raw_scale",
                "raw_clip",
            ],
        ),
        (
            "Optimization",
            [
                "steps",
                "batch_size",
                "num_workers",
                "eval_num_workers",
                "lr",
                "weight_decay",
                "grad_clip",
                "sigma_floor",
            ],
        ),
        (
            "Loss",
            [
                "pred_weight",
                "recon_weight",
                "recon_kind",
                "huber_delta",
                "use_raw_recon_loss",
                "raw_recon_weight",
                "raw_recon_kind",
                "raw_huber_delta",
                "use_anom_weight",
                "anom_weight",
                "identity_loss_weight",
                "cls_weight",
                "cls_pos_weight",
                "cls_label_threshold",
            ],
        ),
        (
            "Sampling",
            [
                "station_uniform",
                "seg_weight_by_len",
                "event_prob",
                "event_future_hours",
                "event_dh_thresh",
                "event_max_tries",
                "recon_k",
                "eval_batches",
            ],
        ),
        (
            "Synthetic Anomaly",
            [
                "use_synth_anomaly",
                "disable_local_sigma_label",
                "local_sigma_window",
                "local_sigma_k",
                "local_sigma_single_k",
                "local_sigma_min_std",
                "identity_prob",
                "anom_point_ratio_min",
                "anom_point_ratio_max",
                "anom_local_scale_window",
                "anom_min_local_scale",
                "anom_clip_min",
                "anom_clip_max",
                "anom_max_events",
                "anom_max_tries_per_event",
            ],
        ),
        (
            "Spike",
            [
                "spike_prob",
                "spike_count_min",
                "spike_count_max",
                "spike_width_min",
                "spike_width_max",
                "spike_amp_k_min",
                "spike_amp_k_max",
            ],
        ),
        (
            "Oscillation",
            [
                "osc_prob",
                "osc_dur_min",
                "osc_dur_max",
                "osc_amp_k_min",
                "osc_amp_k_max",
                "osc_period_min",
                "osc_period_max",
                "osc_noise_ratio",
            ],
        ),
        (
            "Ramp",
            [
                "ramp_prob",
                "ramp_dur_min",
                "ramp_dur_max",
                "ramp_amp_k_min",
                "ramp_amp_k_max",
                "ramp_step_prob",
            ],
        ),
        (
            "Logging And Eval",
            [
                "log_every",
                "ckpt_every",
                "fast_eval_every",
                "fast_eval_batches",
                "fast_eval_recon_patches",
                "full_eval_every",
                "full_eval_batches",
                "early_stop_patience",
                "early_stop_min_delta",
                "early_stop_metric",
                "early_stop_warmup_steps",
            ],
        ),
    ]

    consumed = set()
    lines = ["# Hyperparameters", ""]
    for title, keys in groups:
        lines.append(f"## {title}")
        lines.append("")
        for key in keys:
            if key in args_dict:
                lines.append(f"- `{key}`: `{args_dict[key]}`")
                consumed.add(key)
        lines.append("")

    remaining = sorted(set(args_dict.keys()) - consumed)
    if remaining:
        lines.append("## Other")
        lines.append("")
        for key in remaining:
            lines.append(f"- `{key}`: `{args_dict[key]}`")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_checkpoint_light(
    out_dir: str,
    tag: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    extra: Dict[str, Any],
    update_latest_index: bool = False,
) -> str:
    _ensure_dir(out_dir)
    ckpt_dir = os.path.join(out_dir, f"{tag}_step_{step:06d}")
    _ensure_dir(ckpt_dir)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_full.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    meta = {"tag": tag, "step": step, "extra": extra}
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if update_latest_index:
        with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump({"latest": os.path.basename(ckpt_dir)}, f, ensure_ascii=False, indent=2)

    print(f"[ckpt] saved: {ckpt_dir}")
    return ckpt_dir


def _cleanup_checkpoints(out_dir: str, tag: str, keep: int) -> None:
    if keep <= 0 or not os.path.isdir(out_dir):
        return

    prefix = f"{tag}_step_"
    ckpt_dirs = []
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if not os.path.isdir(path) or not name.startswith(prefix):
            continue
        try:
            step = int(name[len(prefix):])
        except ValueError:
            continue
        ckpt_dirs.append((step, path))

    ckpt_dirs.sort(key=lambda item: item[0], reverse=True)
    for _, path in ckpt_dirs[keep:]:
        shutil.rmtree(path, ignore_errors=True)


def _save_checkpoint_managed(
    out_dir: str,
    tag: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    extra: Dict[str, Any],
    keep: int,
    update_latest_index: bool = False,
) -> str:
    ckpt_dir = save_checkpoint_light(
        out_dir=out_dir,
        tag=tag,
        step=step,
        model=model,
        optimizer=optimizer,
        extra=extra,
        update_latest_index=update_latest_index,
    )
    _cleanup_checkpoints(out_dir, tag=tag, keep=keep)
    return ckpt_dir


def _load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    model_path = os.path.join(ckpt_dir, "model_full.pt")
    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
    meta_path = os.path.join(ckpt_dir, "meta.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"missing model checkpoint: {model_path}")
    if not os.path.isfile(optimizer_path):
        raise FileNotFoundError(f"missing optimizer checkpoint: {optimizer_path}")

    model_sd = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if unexpected:
        print(f"[resume] unexpected keys: {unexpected[:8]}")
    if missing:
        print(f"[resume] missing keys count: {len(missing)}")

    optimizer_sd = torch.load(optimizer_path, map_location=device)
    optimizer.load_state_dict(optimizer_sd)

    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    extra = meta.get("extra", {})
    return {
        "step": int(meta.get("step", 0)),
        "best_metric": extra.get("best_metric"),
        "bad_count": int(extra.get("bad_count", 0)),
        "meta": meta,
    }


def _resolve_resume_ckpt(out_dir: str, resume_ckpt: str | None, resume_latest: bool) -> str | None:
    if resume_ckpt:
        return resume_ckpt
    if not resume_latest:
        return None

    latest_path = os.path.join(out_dir, "latest.json")
    if not os.path.isfile(latest_path):
        raise FileNotFoundError(f"missing latest.json: {latest_path}")

    with open(latest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    latest_name = payload.get("latest")
    if not latest_name:
        raise ValueError(f"invalid latest.json: {latest_path}")
    return os.path.join(out_dir, latest_name)


def _cast_trainables_to_fp32(model: torch.nn.Module) -> None:
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()


def _build_anomaly_cfg(args: argparse.Namespace) -> AnomalyInjectConfig:
    return AnomalyInjectConfig(
        enabled=bool(args.use_synth_anomaly),
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


def _expand_batch_selected_recon(batch: Dict[str, Any], qidx: torch.Tensor) -> Dict[str, Any]:
    bsz = batch["stats_vec"].shape[0]
    hist_n = batch["rec_all_hist_z_clean"].shape[1]
    device = batch["stats_vec"].device
    qidx = qidx.to(device=device, dtype=torch.long)
    if qidx.ndim == 1:
        qidx = qidx.unsqueeze(0).expand(bsz, -1)
    recon_k = qidx.shape[1]

    out: Dict[str, Any] = {}
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


def _expand_batch_random_k_recon(batch: Dict[str, Any], recon_k: int) -> Dict[str, Any]:
    if recon_k <= 1:
        return batch

    hist_n = batch["rec_all_hist_z_clean"].shape[1]
    recon_k = min(int(recon_k), int(hist_n))
    device = batch["stats_vec"].device
    qidx = torch.stack(
        [torch.randperm(hist_n, device=device)[:recon_k] for _ in range(batch["stats_vec"].shape[0])],
        dim=0,
    )
    return _expand_batch_selected_recon(batch, qidx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=_DEFAULT_DATA_DIR)
    ap.add_argument("--train_split", type=str, default="train", choices=["train", "eval"])
    ap.add_argument("--eval_split", type=str, default="eval", choices=["train", "eval"])
    ap.add_argument("--out_dir", type=str, default="./ckpt")
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--resume_latest", action="store_true",default=False)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cov_dim", type=int, default=5)

    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--sigma_floor", type=float, default=0.05)

    ap.add_argument("--pred_weight", type=float, default=0.10)
    ap.add_argument("--recon_weight", type=float, default=0.15)
    ap.add_argument("--recon_kind", type=str, default="huber", choices=["huber", "l1", "mse"])
    ap.add_argument("--huber_delta", type=float, default=0.8)

    ap.add_argument("--use_raw_recon_loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--raw_recon_weight", type=float, default=0.55)
    ap.add_argument("--raw_recon_kind", type=str, default="huber", choices=["huber", "l1", "mse"])
    ap.add_argument("--raw_huber_delta", type=float, default=0.4)

    ap.add_argument("--use_anom_weight", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--anom_weight", type=float, default=2.0)
    ap.add_argument("--identity_loss_weight", type=float, default=0.05)
    ap.add_argument("--cls_weight", type=float, default=2.0)
    ap.add_argument("--cls_pos_weight", type=float, default=8.0)
    ap.add_argument("--cls_label_threshold", type=float, default=0.15)

    ap.add_argument("--station_uniform", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seg_weight_by_len", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--event_prob", type=float, default=0.25)
    ap.add_argument("--event_future_hours", type=int, default=3)
    ap.add_argument("--event_dh_thresh", type=float, default=0.5)
    ap.add_argument("--event_max_tries", type=int, default=60)

    ap.add_argument("--use_synth_anomaly", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable_local_sigma_label", action="store_true",default=False)
    ap.add_argument("--local_sigma_window", type=int, default=0)
    ap.add_argument("--local_sigma_k", type=float, default=5.0)
    ap.add_argument("--local_sigma_single_k", type=float, default=6.0)
    ap.add_argument("--local_sigma_min_std", type=float, default=1e-6)
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

    ap.add_argument("--pred_patch_size", type=int, default=12)
    ap.add_argument("--recon_patch_size", type=int, default=6)
    ap.add_argument("--use_raw_branch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--raw_scale", type=float, default=1.0)
    ap.add_argument("--raw_clip", type=float, default=None)

    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)

    ap.add_argument("--fast_eval_every", type=int, default=500)
    ap.add_argument("--fast_eval_batches", type=int, default=120)
    ap.add_argument("--fast_eval_recon_patches", type=int, default=6)
    ap.add_argument("--full_eval_every", type=int, default=1000)
    ap.add_argument("--full_eval_batches", type=int, default=400)
    ap.add_argument("--early_stop_patience", type=int, default=12)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.001)
    ap.add_argument("--recon_k", type=int, default=4)
    ap.add_argument(
        "--early_stop_metric",
        type=str,
        default="cls_pr_auc",
        choices=["cls_best_f1", "cls_pr_auc", "cls_fixed_f1", "cls_fixed_precision"],
    )
    ap.add_argument("--early_stop_warmup_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    _ensure_dir(args.out_dir)
    log_path = os.path.join(args.out_dir, "train.log.jsonl")
    _save_hparams_doc(args.out_dir, args)

    station_uniform = bool(args.station_uniform)
    seg_weight_by_len = bool(args.seg_weight_by_len)
    use_raw_branch = bool(args.use_raw_branch)
    use_raw_recon_loss = bool(args.use_raw_recon_loss)
    use_anom_weight = bool(args.use_anom_weight)

    datasets = load_datasets(args.data_dir)
    gstats = compute_global_norm_stats(datasets["train"])
    anomaly_cfg = _build_anomaly_cfg(args)

    ds_train = X3TwoPassDataset(
        split_data=datasets[args.train_split],
        global_stats=gstats,
        samples_per_epoch=max(args.steps * args.batch_size, 2000),
        seed=args.seed,
        deterministic_rec_hour=True,
        sigma_floor=args.sigma_floor,
        station_uniform=station_uniform,
        seg_weight_by_len=seg_weight_by_len,
        event_prob=args.event_prob,
        event_future_hours=args.event_future_hours,
        event_dh_thresh=args.event_dh_thresh,
        event_max_tries=args.event_max_tries,
        use_synth_anomaly=bool(args.use_synth_anomaly),
        anomaly_cfg=anomaly_cfg,
        local_sigma_enabled=not bool(args.disable_local_sigma_label),
        local_sigma_window=args.local_sigma_window,
        local_sigma_k=args.local_sigma_k,
        local_sigma_single_k=args.local_sigma_single_k,
        local_sigma_min_std=args.local_sigma_min_std,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_two_pass,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    ds_eval = X3TwoPassDataset(
        split_data=datasets[args.eval_split],
        global_stats=gstats,
        samples_per_epoch=max(args.full_eval_batches * args.batch_size, 2000),
        seed=args.seed + 123,
        deterministic_rec_hour=True,
        sigma_floor=args.sigma_floor,
        station_uniform=True,
        seg_weight_by_len=True,
        event_prob=0.0,
        event_future_hours=args.event_future_hours,
        event_dh_thresh=args.event_dh_thresh,
        event_max_tries=args.event_max_tries,
        use_synth_anomaly=True,
        anomaly_cfg=anomaly_cfg,
        local_sigma_enabled=not bool(args.disable_local_sigma_label),
        local_sigma_window=args.local_sigma_window,
        local_sigma_k=args.local_sigma_k,
        local_sigma_single_k=args.local_sigma_single_k,
        local_sigma_min_std=args.local_sigma_min_std,
    )
    dl_eval = DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        collate_fn=collate_two_pass,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.eval_num_workers > 0),
    )

    model = DPMTFormer(
        DPMTFormerConfig(
            backbone_path="pmtformer",
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

    for _, param in model.named_parameters():
        param.requires_grad = True

    _cast_trainables_to_fp32(model)

    print("[params]", _count_trainable_params(model))

    _append_jsonl(
        log_path,
        {
            "type": "meta",
            "args": vars(args),
            "global_stats": gstats.__dict__,
            "anomaly_cfg": {
                "enabled": anomaly_cfg.enabled,
                "identity_prob": anomaly_cfg.identity_prob,
                "point_ratio_min": anomaly_cfg.point_ratio_min,
                "point_ratio_max": anomaly_cfg.point_ratio_max,
            },
        },
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
            min_lr_ratio=float(args.min_lr) / float(args.lr),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_cfg = LossConfig(
        pred_weight=args.pred_weight,
        recon_weight=args.recon_weight,
        recon_kind=args.recon_kind,
        huber_delta=args.huber_delta,
        use_raw_recon_loss=use_raw_recon_loss,
        raw_recon_weight=args.raw_recon_weight,
        raw_recon_kind=args.raw_recon_kind,
        raw_huber_delta=args.raw_huber_delta,
        use_anom_weight=use_anom_weight,
        anom_weight=args.anom_weight,
        identity_loss_weight=args.identity_loss_weight,
        cls_weight=args.cls_weight,
        cls_pos_weight=args.cls_pos_weight,
        cls_label_threshold=args.cls_label_threshold,
    )

    max_metrics = {"cls_best_f1", "cls_pr_auc", "cls_fixed_f1", "cls_fixed_precision"}
    best_metric = float("-inf") if args.early_stop_metric in max_metrics else float("inf")
    bad_count = 0
    resume_step = 0
    resume_target = _resolve_resume_ckpt(args.out_dir, args.resume_ckpt, args.resume_latest)

    if resume_target:
        resume_state = _load_resume_checkpoint(model, optimizer, resume_target, device)
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_build_lr_lambda(
                warmup_steps=args.warmup_steps,
                total_steps=args.steps,
                min_lr_ratio=float(args.min_lr) / float(args.lr),
            ),
            last_epoch=max(int(resume_state["step"]) - 1, -1),
        )
        resume_step = int(resume_state["step"])
        resumed_best_metric = resume_state.get("best_metric")
        if resumed_best_metric is not None:
            best_metric = float(resumed_best_metric)
        bad_count = int(resume_state.get("bad_count", 0))
        print(
            f"[resume] ckpt={resume_target} step={resume_step} "
            f"best_metric={best_metric:.6f} bad_count={bad_count}"
        )
        if resume_step >= args.steps:
            raise ValueError(f"resume step {resume_step} >= target steps {args.steps}")

    model.train()
    t_start = time.time()

    data_iter = iter(dl_train)

    for step in range(resume_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl_train)
            batch = next(data_iter)

        batch = _move_batch_to_device(batch, device)
        batch = {key: value for key, value in batch.items() if key != "station_id"}
        batch_k = _expand_batch_random_k_recon(batch, recon_k=int(args.recon_k))

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out_pred = model.forward_pred(batch)
            out_rec = model.forward_recon(batch_k)
            out_cls = model.forward_cls(batch)
            out = {**out_pred, **out_rec, **out_cls}
            if "rec_query_anom_ratio" in batch_k:
                out["rec_query_anom_ratio"] = batch_k["rec_query_anom_ratio"]
            if "rec_is_identity" in batch_k:
                out["rec_is_identity"] = batch_k["rec_is_identity"]
            losses = compute_losses(out, loss_cfg)
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if (not use_amp) or (scaler.get_scale() >= prev_scale):
            scheduler.step()

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_start
            sdiag = _sigma_diag_for_console(out)
            current_lr = float(optimizer.param_groups[0]["lr"])
            rec_anom_ratio = (
                float(batch["rec_anom_point_ratio"].detach().float().mean().cpu())
                if "rec_anom_point_ratio" in batch
                else float("nan")
            )
            rec_identity_ratio = (
                float(batch["rec_is_identity"].detach().float().mean().cpu())
                if "rec_is_identity" in batch
                else float("nan")
            )

            payload = {
                "type": "train",
                "step": step,
                "recon_k": args.recon_k,
                "sec": round(dt, 3),
                "lr": current_lr,
                "loss_total": float(losses["loss_total"].detach().float().cpu()),
                "loss_pred_nll": float(losses["loss_pred_nll"].detach().float().cpu()),
                "loss_recon": float(losses["loss_recon"].detach().float().cpu()),
                "loss_recon_z": float(losses["loss_recon_z"].detach().float().cpu()),
                "loss_recon_raw": float(losses["loss_recon_raw"].detach().float().cpu()),
                "loss_identity": float(losses["loss_identity"].detach().float().cpu()),
                "loss_cls": float(losses["loss_cls"].detach().float().cpu()),
                "batch_anom_ratio": rec_anom_ratio,
                "batch_identity_ratio": rec_identity_ratio,
                "sigma_min": float(out["pred_params"]["sigma"].detach().float().cpu().min()),
                "sigma_p50": float(out["pred_params"]["sigma"].detach().float().cpu().median()),
                "sigma_p90": float(torch.quantile(out["pred_params"]["sigma"].detach().float().cpu().reshape(-1), 0.9)),
                "sigma_max": float(out["pred_params"]["sigma"].detach().float().cpu().max()),
            }
            _append_jsonl(log_path, payload)
            print(
                f"[train] step={step:6d} "
                f"loss={float(losses['loss_total'].detach().float().cpu()):.4f} "
                f"rec_k{args.recon_k}={float(losses['loss_recon'].detach().float().cpu()):.4f} "
                f"rz={float(losses['loss_recon_z'].detach().float().cpu()):.4f} "
                f"rr={float(losses['loss_recon_raw'].detach().float().cpu()):.4f} "
                f"id={float(losses['loss_identity'].detach().float().cpu()):.4f} "
                f"cls={float(losses['loss_cls'].detach().float().cpu()):.4f} "
                f"pred={float(losses['loss_pred_nll'].detach().float().cpu()):.4f} "
                f"anom={rec_anom_ratio:.3f} iden={rec_identity_ratio:.3f} "
                f"sig50={sdiag['sig50']:.3f} lr={current_lr:.7f} sec={dt:.1f}"
            )

        run_fast_eval = args.fast_eval_every > 0 and step % args.fast_eval_every == 0
        run_full_eval = args.full_eval_every > 0 and step % args.full_eval_every == 0

        if run_fast_eval or run_full_eval:
            ds_eval.reset_rng(args.seed + 123)
            fast_metrics = None

            if run_fast_eval:
                fast_metrics = eval_model(
                    model=model,
                    dl=dl_eval,
                    device=device,
                    batches=args.fast_eval_batches,
                    recon_kind=args.recon_kind,
                    huber_delta=args.huber_delta,
                    raw_recon_kind=args.raw_recon_kind,
                    raw_huber_delta=args.raw_huber_delta,
                    eval_with_synth_anomaly=True,
                    cls_label_threshold=args.cls_label_threshold,
                    recon_eval_patches=args.fast_eval_recon_patches,
                )
                fast_metrics["step"] = float(step)
                _append_jsonl(log_path, {"type": "eval_fast", **fast_metrics})
                print(
                    f"[evalf] step={step:6d} "
                    f"patches={int(fast_metrics['recon_eval_patches'])} pr_auc={fast_metrics['cls_pr_auc']:.4f} "
                    f"cls_f1={fast_metrics['cls_best_f1']:.4f} rec={fast_metrics['rec_mean']:.4f} "
                    f"pred_nll={fast_metrics['pred_nll_mean']:.4f}"
                )

            if run_full_eval:
                ds_eval.reset_rng(args.seed + 123)
                metrics = eval_model(
                    model=model,
                    dl=dl_eval,
                    device=device,
                    batches=args.full_eval_batches,
                    recon_kind=args.recon_kind,
                    huber_delta=args.huber_delta,
                    raw_recon_kind=args.raw_recon_kind,
                    raw_huber_delta=args.raw_huber_delta,
                    eval_with_synth_anomaly=True,
                    cls_label_threshold=args.cls_label_threshold,
                    recon_eval_patches=N_RECON_HIST_PATCH,
                )
                metrics["step"] = float(step)
                _append_jsonl(log_path, {"type": "eval_full", **metrics})

                cls_f1 = metrics.get("cls_best_f1", float("nan"))
                cls_prec = metrics.get("cls_best_precision", float("nan"))
                cls_rec = metrics.get("cls_best_recall", float("nan"))
                cls_pr_auc = metrics.get("cls_pr_auc", float("nan"))
                cls_fixed_f1 = metrics.get("cls_fixed_f1", float("nan"))
                cls_fixed_prec = metrics.get("cls_fixed_precision", float("nan"))
                print(
                    f"[eval ] step={step:6d} "
                    f"pr_auc={cls_pr_auc:.4f} cls_f1={cls_f1:.4f} cls_p={cls_prec:.4f} cls_r={cls_rec:.4f} "
                    f"fix_f1={cls_fixed_f1:.4f} fix_p={cls_fixed_prec:.4f} "
                    f"rec={metrics['rec_mean']:.4f} recz={metrics['rec_z_mean']:.4f} recr={metrics['rec_raw_mean']:.4f} "
                    f"pred_nll={metrics['pred_nll_mean']:.4f} ae90={metrics['abs_err_p90']:.4f}"
                )

                current = float(metrics.get(args.early_stop_metric, float("nan")))
                if math.isnan(current):
                    improved = False
                elif args.early_stop_metric in max_metrics:
                    improved = (current - best_metric) > args.early_stop_min_delta
                else:
                    improved = (best_metric - current) > args.early_stop_min_delta

                if improved:
                    best_metric = current
                    bad_count = 0
                    _save_checkpoint_managed(
                        args.out_dir,
                        tag="best",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        extra={
                            "best_metric": best_metric,
                            "metric_name": args.early_stop_metric,
                            "eval": metrics,
                            "args": vars(args),
                            "global_stats": gstats.__dict__,
                        },
                        keep=2,
                    )
                else:
                    if step >= args.early_stop_warmup_steps:
                        bad_count += 1
                        print(
                            f"[early-stop] no improve: bad_count={bad_count}/{args.early_stop_patience} "
                            f"(best={best_metric:.6f}, current={current:.6f})"
                        )
                    else:
                        print(f"[early-stop] warmup: step={step} < {args.early_stop_warmup_steps}")

                if step >= args.early_stop_warmup_steps and bad_count >= args.early_stop_patience:
                    print(f"[early-stop] STOP at step={step} best={best_metric:.6f}")
                    break

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            _save_checkpoint_managed(
                args.out_dir,
                tag="latest",
                step=step,
                model=model,
                optimizer=optimizer,
                extra={"args": vars(args), "global_stats": gstats.__dict__},
                keep=1,
                update_latest_index=True,
            )

    _save_checkpoint_managed(
        args.out_dir,
        tag="latest",
        step=step,
        model=model,
        optimizer=optimizer,
        extra={"args": vars(args), "best_metric": best_metric, "bad_count": bad_count},
        keep=1,
        update_latest_index=True,
    )


if __name__ == "__main__":
    main()
