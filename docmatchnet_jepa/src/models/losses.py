"""Loss functions for JEPA and original DocMatchNet training."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(config: Any, *keys: str, default: Any = None) -> Any:
    node = config
    for key in keys:
        if isinstance(node, dict):
            if key not in node:
                return default
            node = node[key]
        else:
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
    return node


# ============================================================
# InfoNCE Loss (for JEPA)
# ============================================================
class InfoNCELoss(nn.Module):
    """
    Bi-directional InfoNCE loss as used in VL-JEPA.

    Combines:
    1. Alignment: predicted embedding close to target
    2. Uniformity: push different embeddings apart (anti-collapse)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        temperature: torch.Tensor | float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            predicted_embeddings: (batch, dim) - predicted ideal doctors
            target_embeddings: (batch, dim) - actual matched doctors
            temperature: scalar (learnable)

        Returns:
            loss: scalar
            accuracy: fraction of correct matches (for monitoring)
        """
        pred_norm = F.normalize(predicted_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)

        if not torch.is_tensor(temperature):
            temperature = torch.tensor(float(temperature), device=pred_norm.device, dtype=pred_norm.dtype)
        temperature = torch.clamp(temperature, min=self.eps)

        logits = pred_norm @ target_norm.T / temperature
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.T, labels)
        loss = (loss_p2t + loss_t2p) / 2.0

        with torch.no_grad():
            pred_correct = (logits.argmax(dim=1) == labels).float().mean()
            target_correct = (logits.T.argmax(dim=1) == labels).float().mean()
            accuracy = ((pred_correct + target_correct) / 2.0).item()

        return loss, accuracy


# ============================================================
# VICReg Gate Regularization
# ============================================================
class VICRegGateLoss(nn.Module):
    """
    VICReg-inspired regularization for gate activations.

    Ensures gates are:
    1. Varied across the batch (not collapsed to same value)
    2. Decorrelated (each gate dimension is informative)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, gates_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gates_dict: {
                'clinical': (batch, gate_dim),
                'pastwork': (batch, gate_dim),
                ...
            }
        """
        total_var_loss = torch.tensor(0.0, device=next(iter(gates_dict.values())).device)
        total_cov_loss = torch.tensor(0.0, device=next(iter(gates_dict.values())).device)

        for gate_vals in gates_dict.values():
            gate_vals = gate_vals.float()
            batch_size = gate_vals.shape[0]

            var = gate_vals.var(dim=0, unbiased=False)
            var_loss = F.relu(1.0 - var).mean()
            total_var_loss = total_var_loss + var_loss

            if gate_vals.shape[1] > 1 and batch_size > 1:
                centered = gate_vals - gate_vals.mean(dim=0, keepdim=True)
                cov = (centered.T @ centered) / max(batch_size - 1, 1)
                off_diag = cov - torch.diag(torch.diag(cov))
                cov_loss = (off_diag.pow(2).sum()) / gate_vals.shape[1]
                total_cov_loss = total_cov_loss + cov_loss

        n_gates = max(len(gates_dict), 1)
        return (total_var_loss + total_cov_loss) / n_gates


# ============================================================
# Combined JEPA Loss
# ============================================================
class DocMatchJEPALoss(nn.Module):
    """
    Combined loss for DocMatchNet-JEPA training.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.lambda_infonce = float(_cfg_get(config, "loss", "lambda_infonce", default=1.0))
        self.lambda_gate = float(_cfg_get(config, "loss", "lambda_gate_vicreg", default=0.05))
        self.lambda_rank = float(_cfg_get(config, "loss", "lambda_ranking", default=0.0))

        self.infonce = InfoNCELoss()
        self.vicreg = VICRegGateLoss()

    def forward(self, model_output: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_output: dict from DocMatchNetJEPA.forward()
        """
        infonce_loss, infonce_acc = self.infonce(
            model_output["predicted_ideal"],
            model_output["doctor_embedding"],
            model_output["temperature"],
        )

        gate_loss = self.vicreg(model_output["gates"])
        rank_loss = torch.tensor(0.0, device=infonce_loss.device)

        total_loss = self.lambda_infonce * infonce_loss + self.lambda_gate * gate_loss + self.lambda_rank * rank_loss

        temperature = model_output["temperature"]
        temp_val = float(temperature.item() if torch.is_tensor(temperature) else temperature)

        metrics = {
            "total_loss": float(total_loss.item()),
            "infonce_loss": float(infonce_loss.item()),
            "infonce_accuracy": float(infonce_acc),
            "gate_reg_loss": float(gate_loss.item()),
            "ranking_loss": float(rank_loss.item()),
            "temperature": temp_val,
        }
        return total_loss, metrics


# ============================================================
# Original DocMatchNet Loss (Pairwise Ranking)
# ============================================================
class DocMatchOriginalLoss(nn.Module):
    """
    Loss for original DocMatchNet: ranking + BCE + gate entropy.
    """

    def __init__(
        self,
        margin: float = 0.1,
        lambda_rank: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_gate: float = 0.01,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.lambda_bce = lambda_bce
        self.lambda_gate = lambda_gate

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        gate_means: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        rank_loss = F.relu(self.margin - (pos_score - neg_score)).mean()

        bce_loss = (
            F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
            + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
        ) / 2.0

        gate_entropy = torch.tensor(0.0, device=pos_score.device)
        eps = 1e-8
        for gate_vals in gate_means.values():
            entropy = -(
                gate_vals * torch.log(gate_vals + eps)
                + (1.0 - gate_vals) * torch.log(1.0 - gate_vals + eps)
            )
            gate_entropy = gate_entropy + entropy.mean()
        gate_entropy = gate_entropy / max(len(gate_means), 1)

        total_loss = (
            self.lambda_rank * rank_loss
            + self.lambda_bce * bce_loss
            + self.lambda_gate * gate_entropy
        )

        return total_loss, {
            "total_loss": float(total_loss.item()),
            "rank_loss": float(rank_loss.item()),
            "bce_loss": float(bce_loss.item()),
            "gate_entropy": float(gate_entropy.item()),
        }
