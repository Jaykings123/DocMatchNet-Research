"""Original DocMatchNet score-space architecture."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _cfg_get(config: Any, *keys: str, default: Any = None) -> Any:
    """Read nested config values from dict-like or object-like configs."""
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


class DocMatchNetOriginal(nn.Module):
    """
    Original DocMatchNet with score-space prediction.

    This predicts a scalar match score directly,
    unlike JEPA which predicts embeddings.
    """

    def __init__(self, config: Any):
        super().__init__()

        embed_dim = int(_cfg_get(config, "model", "embedding_dim", default=384))
        hidden_dim = int(_cfg_get(config, "model", "latent_dim", default=256))
        gate_dim = int(_cfg_get(config, "model", "gate_dim", default=32))
        n_heads = int(_cfg_get(config, "model", "n_attention_heads", default=4))
        dropout = float(_cfg_get(config, "model", "dropout", default=0.1))
        context_dim = 8

        # ========== Embedding Interaction (Cross-Attention) ==========
        self.patient_proj = nn.Linear(embed_dim, hidden_dim)
        self.doctor_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.interaction_norm = nn.LayerNorm(hidden_dim)

        # ========== Dimension Encoders ==========
        self.clinical_encoder = self._make_encoder(4, gate_dim)
        self.pastwork_encoder = self._make_encoder(5, gate_dim)
        self.logistics_encoder = self._make_encoder(5, gate_dim)
        self.trust_encoder = self._make_encoder(3, gate_dim)

        # ========== Context-Aware Gates ==========
        gate_input_dim = hidden_dim + context_dim
        self.clinical_gate = self._make_gate(gate_input_dim, gate_dim)
        self.pastwork_gate = self._make_gate(gate_input_dim, gate_dim)
        self.logistics_gate = self._make_gate(gate_input_dim, gate_dim)
        self.trust_gate = self._make_gate(gate_input_dim, gate_dim)
        self._init_gate_biases()

        # ========== Scoring Head (DIFFERENT FROM JEPA) ==========
        self.scoring_head = nn.Sequential(
            nn.Linear(gate_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _make_encoder(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def _make_gate(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def _init_gate_biases(self) -> None:
        nn.init.constant_(self.clinical_gate[-2].bias, 0.4)
        nn.init.constant_(self.pastwork_gate[-2].bias, 0.0)
        nn.init.constant_(self.logistics_gate[-2].bias, 0.0)
        nn.init.constant_(self.trust_gate[-2].bias, -0.4)

    def forward(
        self,
        patient_emb: torch.Tensor,
        doctor_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
        pastwork_feat: torch.Tensor,
        logistics_feat: torch.Tensor,
        trust_feat: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """
        Forward pass with score-space prediction.
        """
        p = self.patient_proj(patient_emb).unsqueeze(1)
        d = self.doctor_proj(doctor_emb).unsqueeze(1)

        interaction, attn_weights = self.cross_attention(p, d, d)
        interaction = self.interaction_norm(interaction.squeeze(1))

        enc_clinical = self.clinical_encoder(clinical_feat)
        enc_pastwork = self.pastwork_encoder(pastwork_feat)
        enc_logistics = self.logistics_encoder(logistics_feat)
        enc_trust = self.trust_encoder(trust_feat)

        gate_input = torch.cat([interaction, context], dim=-1)
        g_clinical = self.clinical_gate(gate_input)
        g_pastwork = self.pastwork_gate(gate_input)
        g_logistics = self.logistics_gate(gate_input)
        g_trust = self.trust_gate(gate_input)

        gated = torch.cat(
            [
                g_clinical * enc_clinical,
                g_pastwork * enc_pastwork,
                g_logistics * enc_logistics,
                g_trust * enc_trust,
            ],
            dim=-1,
        )

        score = self.scoring_head(gated)

        gate_means = {
            "clinical": g_clinical.mean(dim=-1),
            "pastwork": g_pastwork.mean(dim=-1),
            "logistics": g_logistics.mean(dim=-1),
            "trust": g_trust.mean(dim=-1),
        }

        return {
            "score": score,
            "gate_means": gate_means,
            "attention_weights": attn_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
