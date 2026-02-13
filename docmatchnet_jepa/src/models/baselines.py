"""Baseline models and ablations for DocMatchNet experiments."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .docmatchnet_jepa import DocMatchNetJEPA


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
# BASELINE 1: Static MCDA (No Learning)
# ============================================================
class StaticMCDA:
    """
    Fixed-weight multi-criteria scoring.
    No trainable parameters - pure formula.
    """

    def __init__(self) -> None:
        self.weights = {
            "clinical": 0.40,
            "pastwork": 0.25,
            "logistics": 0.25,
            "trust": 0.10,
        }
        self.clinical_weights = [0.55, 0.20, 0.15, 0.10]
        self.pastwork_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        self.logistics_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        self.trust_weights = [0.50, 0.30, 0.20]

    def score(
        self,
        clinical: torch.Tensor,
        pastwork: torch.Tensor,
        logistics: torch.Tensor,
        trust: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MCDA score for a batch."""
        device = clinical.device
        c_w = torch.tensor(self.clinical_weights, dtype=clinical.dtype, device=device)
        p_w = torch.tensor(self.pastwork_weights, dtype=pastwork.dtype, device=device)
        l_w = torch.tensor(self.logistics_weights, dtype=logistics.dtype, device=device)
        t_w = torch.tensor(self.trust_weights, dtype=trust.dtype, device=device)

        c_score = (clinical * c_w).sum(dim=-1)
        p_score = (pastwork * p_w).sum(dim=-1)
        l_score = (logistics * l_w).sum(dim=-1)
        t_score = (trust * t_w).sum(dim=-1)

        return (
            self.weights["clinical"] * c_score
            + self.weights["pastwork"] * p_score
            + self.weights["logistics"] * l_score
            + self.weights["trust"] * t_score
        )


# ============================================================
# BASELINE 2: Simple MLP
# ============================================================
class SimpleMLP(nn.Module):
    """
    Concatenate all features -> MLP -> score.
    No structure, no gating, no attention.
    """

    def __init__(self, config: Any):
        super().__init__()
        embed_dim = int(_cfg_get(config, "model", "embedding_dim", default=384))
        input_dim = embed_dim + embed_dim + 4 + 5 + 5 + 3 + 8

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        doctor_emb: torch.Tensor,
        clinical: torch.Tensor,
        pastwork: torch.Tensor,
        logistics: torch.Tensor,
        trust: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([patient_emb, doctor_emb, clinical, pastwork, logistics, trust, context], dim=-1)
        return {"score": self.network(x)}


# ============================================================
# BASELINE 3: Neural Ranker (Cross-Encoder)
# ============================================================
class NeuralRanker(nn.Module):
    """
    Cross-encoder that processes patient and doctor together.
    Uses attention but no gating mechanism.
    """

    def __init__(self, config: Any):
        super().__init__()
        embed_dim = int(_cfg_get(config, "model", "embedding_dim", default=384))
        hidden_dim = 256
        num_heads = int(_cfg_get(config, "model", "n_attention_heads", default=4))

        self.patient_proj = nn.Linear(embed_dim, hidden_dim)
        self.doctor_proj = nn.Linear(embed_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.feature_encoder = nn.Sequential(
            nn.Linear(17 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        doctor_emb: torch.Tensor,
        clinical: torch.Tensor,
        pastwork: torch.Tensor,
        logistics: torch.Tensor,
        trust: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p = self.patient_proj(patient_emb).unsqueeze(1)
        d = self.doctor_proj(doctor_emb).unsqueeze(1)

        interaction, _ = self.cross_attention(p, d, d)
        interaction = self.layer_norm(interaction.squeeze(1))

        features = torch.cat([clinical, pastwork, logistics, trust, context], dim=-1)
        feat_enc = self.feature_encoder(features)

        combined = torch.cat([interaction, feat_enc], dim=-1)
        return {"score": self.scorer(combined)}


# ============================================================
# BASELINE 4: DIN (Deep Interest Network)
# ============================================================
class DINModel(nn.Module):
    """
    Attention-based model inspired by Deep Interest Network.
    Uses case as query to attend over doctor features.
    """

    def __init__(self, config: Any):
        super().__init__()
        embed_dim = int(_cfg_get(config, "model", "embedding_dim", default=384))

        self.case_encoder = nn.Linear(embed_dim + 8, 128)
        self.doctor_encoder = nn.Linear(embed_dim + 17, 128)

        self.attention = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patient_emb: torch.Tensor,
        doctor_emb: torch.Tensor,
        clinical: torch.Tensor,
        pastwork: torch.Tensor,
        logistics: torch.Tensor,
        trust: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        case_input = torch.cat([patient_emb, context], dim=-1)
        case_enc = F.relu(self.case_encoder(case_input))

        features = torch.cat([clinical, pastwork, logistics, trust], dim=-1)
        doctor_input = torch.cat([doctor_emb, features], dim=-1)
        doctor_enc = F.relu(self.doctor_encoder(doctor_input))

        attn_input = torch.cat([case_enc, doctor_enc, case_enc * doctor_enc], dim=-1)
        attn_weight = torch.sigmoid(self.attention(attn_input))
        weighted = attn_weight * doctor_enc

        return {"score": self.scorer(weighted)}


# ============================================================
# ABLATION: DocMatchNet-JEPA without Gates
# ============================================================
class DocMatchNetJEPANoGates(DocMatchNetJEPA):
    """
    JEPA architecture but WITHOUT context-aware gates.
    All gates fixed at 0.5 (uniform weighting).

    This isolates the contribution of gating mechanism.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.gate_dim = int(_cfg_get(config, "model", "gate_dim", default=32))

    def compute_gates(self, patient_latent: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        del context
        batch_size = patient_latent.shape[0]
        g_fixed = torch.full(
            (batch_size, self.gate_dim),
            0.5,
            dtype=patient_latent.dtype,
            device=patient_latent.device,
        )
        return {
            "clinical": g_fixed,
            "pastwork": g_fixed,
            "logistics": g_fixed,
            "trust": g_fixed,
        }
