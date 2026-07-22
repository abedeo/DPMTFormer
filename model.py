from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPMTFormerConfig:
    sigma_min: float = 1e-3
    fixed_nu: float = 8.0

    n_hist_patch: int = 12
    n_fut_patch: int = 3

    pred_patch_size: int = 12
    recon_patch_size: int = 6
    max_cls_pos: int = 32

    max_patch_pos: int = 32
    max_recon_pos: int = 64

    use_raw_branch: bool = True
    raw_patch_dim: int = 12
    raw_recon_patch_dim: int = 6
    raw_scale: float = 1.0
    raw_clip: Optional[float] = None

    cov_dim: int = 5
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 2048
    dropout: float = 0.1


class _PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if pos is None:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return x + self.emb(pos)


class DPMTFormer(nn.Module):
    def __init__(self, cfg: DPMTFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        def _make_encoder() -> nn.TransformerEncoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            return nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.backbone = _make_encoder()
        self.cls_backbone = _make_encoder()

        pred_in_dim = cfg.pred_patch_size + cfg.cov_dim
        rec_in_dim = cfg.recon_patch_size + cfg.cov_dim
        if cfg.use_raw_branch:
            pred_in_dim += cfg.raw_patch_dim
            rec_in_dim += cfg.raw_recon_patch_dim

        self.pred_in_proj = nn.Linear(pred_in_dim, d)
        self.recon_in_proj = nn.Linear(rec_in_dim, d)
        self.cls_in_proj = nn.Linear(rec_in_dim, d)

        rec_query_in_dim = cfg.cov_dim + (cfg.raw_recon_patch_dim if cfg.use_raw_branch else 0)
        self.recon_query_proj = nn.Linear(rec_query_in_dim, d)
        self.recon_query_pos_emb = nn.Embedding(cfg.max_recon_pos, d)

        self.pos_emb_pred = _PositionalEmbedding(cfg.max_patch_pos, d)
        self.pos_emb_recon = _PositionalEmbedding(cfg.max_recon_pos, d)
        self.pos_emb_cls = _PositionalEmbedding(cfg.max_cls_pos, d)

        self.pred_head = nn.Linear(d, cfg.n_fut_patch * cfg.pred_patch_size * 3)
        self.recon_head = nn.Linear(d, cfg.recon_patch_size)
        self.recon_raw_head = nn.Linear(d, cfg.raw_recon_patch_dim) if cfg.use_raw_branch else None

        self.pred_token_proj = nn.Sequential(
            nn.Linear(cfg.pred_patch_size * 3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.pred_cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.recon_aux_proj = nn.Sequential(
            nn.Linear(d + 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.pred_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.recon_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

        self._cached_pred_tokens: Optional[torch.Tensor] = None
        self._cached_recon_aux: Optional[torch.Tensor] = None

    def _maybe_scale_raw(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() * float(self.cfg.raw_scale)
        if self.cfg.raw_clip is not None:
            x = torch.clamp(x, -float(self.cfg.raw_clip), float(self.cfg.raw_clip))
        return x

    def _encode_pred(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = batch["hist_z_patches"].float()
        cov = batch["cov_patches"].float()[:, : self.cfg.n_hist_patch, :]
        feats = [z, cov]
        if self.cfg.use_raw_branch and "hist_raw_patches" in batch:
            feats.append(self._maybe_scale_raw(batch["hist_raw_patches"]))
        x = torch.cat(feats, dim=-1)
        x = self.pred_in_proj(x)
        x = self.pos_emb_pred(x)
        return self.backbone(x)

    def _encode_recon(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = batch["rec_visible_z_patches"].float()
        cov = batch["rec_visible_cov_patches"].float()
        feats = [z, cov]
        if self.cfg.use_raw_branch and "rec_visible_raw_patches" in batch:
            feats.append(self._maybe_scale_raw(batch["rec_visible_raw_patches"]))
        x = torch.cat(feats, dim=-1)
        x = self.recon_in_proj(x)
        x = self.pos_emb_recon(x)
        return self.backbone(x)

    def _encode_cls(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hist_z = batch["rec_all_hist_z_noisy"].float()
        hist_cov = batch["rec_all_hist_cov"].float()
        fut_z = batch["rec_fut_z_noisy"].float()
        fut_cov = batch["rec_fut_cov"].float()

        feats = [torch.cat([hist_z, fut_z], dim=1), torch.cat([hist_cov, fut_cov], dim=1)]
        if self.cfg.use_raw_branch and "rec_all_hist_raw_noisy" in batch and "rec_fut_raw_noisy" in batch:
            hist_raw = self._maybe_scale_raw(batch["rec_all_hist_raw_noisy"])
            fut_raw = self._maybe_scale_raw(batch["rec_fut_raw_noisy"])
            feats.append(torch.cat([hist_raw, fut_raw], dim=1))

        x = torch.cat(feats, dim=-1)
        x = self.cls_in_proj(x)
        x = self.pos_emb_cls(x)
        return self.cls_backbone(x)

    def forward_pred(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self._encode_pred(batch)
        pooled = h.mean(dim=1)

        pred = self.pred_head(pooled)
        bsz = pred.shape[0]
        n_fut = self.cfg.n_fut_patch
        patch_size = self.cfg.pred_patch_size
        pred = pred.view(bsz, n_fut, patch_size, 3)

        mu = pred[..., 0]
        sigma = F.softplus(pred[..., 1]) + float(self.cfg.sigma_min)
        nu = torch.full_like(mu, float(self.cfg.fixed_nu))

        pred_tokens = torch.cat([mu, sigma, nu], dim=-1)
        self._cached_pred_tokens = self.pred_token_proj(pred_tokens)

        return {
            "pred_params": {"mu": mu, "sigma": sigma, "nu": nu},
            "target_fut_z": batch["fut_z_patches"].float(),
        }

    def forward_recon(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self._encode_recon(batch)
        q_cov = batch["rec_query_cov"].float()
        q_feats = [q_cov]
        if self.cfg.use_raw_branch and "rec_query_raw_patch" in batch:
            q_feats.append(self._maybe_scale_raw(batch["rec_query_raw_patch"]))
        q = torch.cat(q_feats, dim=-1)
        q = self.recon_query_proj(q)
        if "rec_query_pos" in batch:
            q_pos = batch["rec_query_pos"].long()
            q = q + self.recon_query_pos_emb(q_pos)

        ctx = h.mean(dim=1)
        fuse = ctx + q
        recon_patch_hat = self.recon_head(fuse)

        out: Dict[str, torch.Tensor] = {
            "recon_patch_hat": recon_patch_hat,
            "target_rec_patch": batch["rec_target_patch"].float(),
        }

        if self.cfg.use_raw_branch and self.recon_raw_head is not None and "rec_target_raw_patch" in batch:
            recon_raw_patch_hat = self.recon_raw_head(fuse)
            out["recon_raw_patch_hat"] = recon_raw_patch_hat
            out["target_rec_raw_patch"] = batch["rec_target_raw_patch"].float()
            raw_err = (recon_raw_patch_hat - out["target_rec_raw_patch"]).abs().mean(dim=-1, keepdim=True)
        else:
            raw_err = torch.zeros((fuse.shape[0], 1), device=fuse.device, dtype=fuse.dtype)

        z_err = (recon_patch_hat - out["target_rec_patch"]).abs().mean(dim=-1, keepdim=True)
        recon_aux = self.recon_aux_proj(torch.cat([fuse, z_err, raw_err], dim=-1)).float()

        group_size = int(batch.get("rec_group_size", 1))
        base_bsz = int(batch.get("rec_base_bsz", fuse.shape[0]))
        patch_slots = int(self.cfg.max_cls_pos)
        recon_patch_aux = torch.zeros(
            base_bsz,
            patch_slots,
            recon_aux.shape[-1],
            device=fuse.device,
            dtype=torch.float32,
        )
        recon_patch_cnt = torch.zeros(base_bsz, patch_slots, 1, device=fuse.device, dtype=torch.float32)

        if "rec_mask_patch_idx" in batch:
            patch_idx = batch["rec_mask_patch_idx"].long()
            sample_idx = torch.arange(base_bsz, device=fuse.device).repeat_interleave(group_size)
            valid = (sample_idx < base_bsz) & (patch_idx >= 0) & (patch_idx < patch_slots)
            if valid.any():
                flat_idx = sample_idx[valid] * patch_slots + patch_idx[valid]
                recon_patch_aux.view(-1, recon_aux.shape[-1]).index_add_(0, flat_idx, recon_aux[valid])
                recon_patch_cnt.view(-1, 1).index_add_(
                    0,
                    flat_idx,
                    torch.ones((flat_idx.shape[0], 1), device=fuse.device, dtype=torch.float32),
                )
        self._cached_recon_aux = recon_patch_aux / recon_patch_cnt.clamp_min(1.0)
        return out

    def forward_cls(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self._encode_cls(batch)
        hist_n = batch["rec_all_hist_z_noisy"].shape[1]
        hist_h = h[:, :hist_n, :]

        pred_tokens = self._cached_pred_tokens
        if pred_tokens is None or pred_tokens.shape[0] != hist_h.shape[0]:
            pred_ctx = torch.zeros_like(hist_h)
        else:
            pred_ctx, _ = self.pred_cross_attn(hist_h, pred_tokens, pred_tokens, need_weights=False)

        recon_aux = self._cached_recon_aux
        if recon_aux is None or recon_aux.shape[0] != hist_h.shape[0]:
            recon_token = torch.zeros_like(hist_h)
        else:
            recon_token = recon_aux[:, :hist_n, :]

        pred_gate = self.pred_gate(torch.cat([hist_h, pred_ctx], dim=-1))
        recon_gate = self.recon_gate(torch.cat([hist_h, recon_token], dim=-1))
        fused_hist_h = hist_h + pred_gate * pred_ctx + recon_gate * recon_token

        return {
            "cls_logits": self.cls_head(fused_hist_h).squeeze(-1),
            "cls_target_anom_ratio": batch["rec_all_hist_anom_ratios"].float(),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {**self.forward_pred(batch), **self.forward_recon(batch), **self.forward_cls(batch)}
