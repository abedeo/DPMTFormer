from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import torch
import torch.nn.functional as F


EPS = 1e-6


def det_hard_label(anom_ratio: torch.Tensor, pos_threshold: float) -> torch.Tensor:
    return (anom_ratio.float() >= float(pos_threshold)).to(dtype=torch.int32)


def student_t_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | float,
    eps: float = EPS,
    reduce: str = "mean",
) -> torch.Tensor:
    if isinstance(nu, (float, int)):
        nu_t = torch.tensor(float(nu), device=y.device, dtype=y.dtype)
    else:
        nu_t = nu.to(device=y.device, dtype=y.dtype)

    nu_t = torch.clamp(nu_t, min=2.0 + 1e-6)
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma

    t1 = torch.lgamma((nu_t + 1.0) / 2.0) - torch.lgamma(nu_t / 2.0)
    t2 = -0.5 * torch.log(nu_t * torch.tensor(math.pi, device=y.device, dtype=y.dtype))
    t3 = -torch.log(sigma)
    t4 = -((nu_t + 1.0) / 2.0) * torch.log1p((z * z) / nu_t)
    nll = -(t1 + t2 + t3 + t4)

    if reduce == "none":
        return nll
    if reduce == "sum":
        return nll.sum()
    if reduce == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduce={reduce}")


def recon_loss(
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
    kind: str = "huber",
    huber_delta: float = 1.0,
    reduce: str = "mean",
) -> torch.Tensor:
    if kind == "l1":
        loss = (pred_patch - target_patch).abs()
    elif kind == "mse":
        loss = (pred_patch - target_patch) ** 2
    elif kind == "huber":
        loss = F.smooth_l1_loss(pred_patch, target_patch, beta=huber_delta, reduction="none")
    else:
        raise ValueError(f"unknown recon loss kind={kind}")

    if reduce == "none":
        return loss
    if reduce == "sum":
        return loss.sum()
    if reduce == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduce={reduce}")


def _weighted_mean(loss_elem: torch.Tensor, weight_elem: Optional[torch.Tensor], eps: float = EPS) -> torch.Tensor:
    if weight_elem is None:
        return loss_elem.mean()

    weight = weight_elem
    while weight.ndim < loss_elem.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.to(dtype=loss_elem.dtype, device=loss_elem.device)
    return torch.sum(loss_elem * weight) / torch.sum(weight).clamp_min(eps)


@dataclass
class LossConfig:
    pred_weight: float = 0.5
    recon_weight: float = 0.3
    recon_kind: str = "huber"
    huber_delta: float = 1.0

    use_raw_recon_loss: bool = True
    raw_recon_weight: float = 0.5
    raw_recon_kind: str = "huber"
    raw_huber_delta: float = 0.5

    use_anom_weight: bool = True
    anom_weight: float = 1.5

    identity_loss_weight: float = 0.1

    cls_weight: float = 1.0
    cls_pos_weight: float = 4.0
    cls_label_threshold: float = 0.15

    eps: float = EPS
    pred_reduce: str = "mean"


def compute_losses(model_out: Dict[str, Any], cfg: Optional[LossConfig] = None) -> Dict[str, torch.Tensor]:
    if cfg is None:
        cfg = LossConfig()

    y = model_out["target_fut_z"]
    mu = model_out["pred_params"]["mu"]
    sigma = model_out["pred_params"]["sigma"]
    nu = model_out["pred_params"]["nu"]
    pred_nll = student_t_nll(y=y, mu=mu, sigma=sigma, nu=nu, eps=cfg.eps, reduce=cfg.pred_reduce)

    rec_hat = model_out["recon_patch_hat"]
    rec_tgt = model_out["target_rec_patch"]
    rec_elem = recon_loss(rec_hat, rec_tgt, kind=cfg.recon_kind, huber_delta=cfg.huber_delta, reduce="none")

    rec_query_anom_ratio = model_out.get("rec_query_anom_ratio")
    weight_z = None
    if cfg.use_anom_weight and rec_query_anom_ratio is not None:
        weight_z = 1.0 + (float(cfg.anom_weight) - 1.0) * rec_query_anom_ratio
    rec_z = _weighted_mean(rec_elem, weight_z, eps=cfg.eps)

    rec_raw = torch.zeros((), device=pred_nll.device, dtype=pred_nll.dtype)
    has_raw = (
        cfg.use_raw_recon_loss
        and "recon_raw_patch_hat" in model_out
        and "target_rec_raw_patch" in model_out
    )
    if has_raw:
        rec_raw_elem = recon_loss(
            model_out["recon_raw_patch_hat"],
            model_out["target_rec_raw_patch"],
            kind=cfg.raw_recon_kind,
            huber_delta=cfg.raw_huber_delta,
            reduce="none",
        )
        rec_raw = _weighted_mean(rec_raw_elem, weight_z, eps=cfg.eps)
    else:
        rec_raw_elem = None

    loss_identity = torch.zeros((), device=pred_nll.device, dtype=pred_nll.dtype)
    rec_is_identity = model_out.get("rec_is_identity")
    if rec_is_identity is not None:
        id_mask = (rec_is_identity > 0).to(device=pred_nll.device, dtype=pred_nll.dtype)
        if id_mask.sum() > 0:
            z_id = rec_elem.mean(dim=-1)
            z_id_loss = (z_id * id_mask).sum() / id_mask.sum().clamp_min(cfg.eps)
            if has_raw and rec_raw_elem is not None:
                raw_id = rec_raw_elem.mean(dim=-1)
                raw_id_loss = (raw_id * id_mask).sum() / id_mask.sum().clamp_min(cfg.eps)
            else:
                raw_id_loss = torch.zeros_like(z_id_loss)
            loss_identity = z_id_loss + float(cfg.raw_recon_weight) * raw_id_loss

    loss_recon_total = rec_z + float(cfg.raw_recon_weight) * rec_raw + float(cfg.identity_loss_weight) * loss_identity

    loss_cls = torch.zeros((), device=pred_nll.device, dtype=pred_nll.dtype)
    cls_target_ratio = model_out.get("cls_target_anom_ratio")
    if "cls_logits" in model_out and cls_target_ratio is not None:
        cls_logits = model_out["cls_logits"].to(device=pred_nll.device, dtype=pred_nll.dtype)
        cls_target = (cls_target_ratio >= float(cfg.cls_label_threshold)).to(
            device=pred_nll.device,
            dtype=pred_nll.dtype,
        )
        pos_weight = torch.tensor(float(cfg.cls_pos_weight), device=pred_nll.device, dtype=pred_nll.dtype)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_target, pos_weight=pos_weight)

    loss_total = (
        float(cfg.pred_weight) * pred_nll
        + float(cfg.recon_weight) * loss_recon_total
        + float(cfg.cls_weight) * loss_cls
    )

    return {
        "loss_total": loss_total,
        "loss_pred_nll": pred_nll,
        "loss_recon": loss_recon_total,
        "loss_recon_z": rec_z,
        "loss_recon_raw": rec_raw,
        "loss_identity": loss_identity,
        "loss_cls": loss_cls,
    }
