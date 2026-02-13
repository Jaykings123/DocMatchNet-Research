"""JEPA-style DocMatchNet architecture."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DocMatchNetJEPA(nn.Module):
    """
    DocMatchNet with JEPA-style embedding prediction.

    Key Innovation: Instead of predicting a scalar score,
    predicts the embedding of the ideal doctor, then computes
    match score as similarity between predicted and actual.
    """

    def __init__(self, config: Any):
        super().__init__()

        embed_dim = int(_cfg_get(config, "model", "embedding_dim", default=384))
        latent_dim = int(_cfg_get(config, "model", "latent_dim", default=256))
        gate_dim = int(_cfg_get(config, "model", "gate_dim", default=32))
        dropout = float(_cfg_get(config, "model", "dropout", default=0.1))
        context_dim = 8

        # ========== Patient Encoder (X-Encoder) ==========
        self.patient_encoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ========== Doctor Encoder (Y-Encoder) ==========
        self.doctor_encoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ========== Dimension Encoders ==========
        self.clinical_encoder = self._make_encoder(4, gate_dim)
        self.pastwork_encoder = self._make_encoder(5, gate_dim)
        self.logistics_encoder = self._make_encoder(5, gate_dim)
        self.trust_encoder = self._make_encoder(3, gate_dim)

        # ========== Context-Aware Gates ==========
        gate_input_dim = latent_dim + context_dim
        self.clinical_gate = self._make_gate(gate_input_dim, gate_dim)
        self.pastwork_gate = self._make_gate(gate_input_dim, gate_dim)
        self.logistics_gate = self._make_gate(gate_input_dim, gate_dim)
        self.trust_gate = self._make_gate(gate_input_dim, gate_dim)
        self._init_gate_biases()

        # ========== Predictor (Core JEPA Component) ==========
        predictor_input_dim = latent_dim + gate_dim * 4
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )

        # ========== Projection Heads (for contrastive learning) ==========
        self.predictor_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 128),
        )
        self.doctor_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 128),
        )

        # ========== Learnable Temperature ==========
        init_temp = float(_cfg_get(config, "loss", "infonce_temperature", default=0.07))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temp, dtype=torch.float32)))

    def _make_encoder(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Dimension encoder network."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def _make_gate(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Context-aware gate network."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def _init_gate_biases(self) -> None:
        """Initialize gates to approximate MCDA priors."""
        nn.init.constant_(self.clinical_gate[-2].bias, 0.4)
        nn.init.constant_(self.pastwork_gate[-2].bias, 0.0)
        nn.init.constant_(self.logistics_gate[-2].bias, 0.0)
        nn.init.constant_(self.trust_gate[-2].bias, -0.4)

    def encode_patient(self, patient_emb: torch.Tensor) -> torch.Tensor:
        """Encode patient into latent space."""
        return self.patient_encoder(patient_emb)

    def encode_doctor(self, doctor_emb: torch.Tensor) -> torch.Tensor:
        """Encode doctor into latent space."""
        return self.doctor_encoder(doctor_emb)

    def compute_gates(self, patient_latent: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute context-aware gate activations."""
        gate_input = torch.cat([patient_latent, context], dim=-1)
        g_clinical = self.clinical_gate(gate_input)
        g_pastwork = self.pastwork_gate(gate_input)
        g_logistics = self.logistics_gate(gate_input)
        g_trust = self.trust_gate(gate_input)
        return {
            "clinical": g_clinical,
            "pastwork": g_pastwork,
            "logistics": g_logistics,
            "trust": g_trust,
        }

    def predict_ideal_doctor(
        self,
        patient_latent: torch.Tensor,
        clinical_feat: torch.Tensor,
        pastwork_feat: torch.Tensor,
        logistics_feat: torch.Tensor,
        trust_feat: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict embedding of ideal doctor for this patient.
        This is the core JEPA prediction step.
        """
        enc_clinical = self.clinical_encoder(clinical_feat)
        enc_pastwork = self.pastwork_encoder(pastwork_feat)
        enc_logistics = self.logistics_encoder(logistics_feat)
        enc_trust = self.trust_encoder(trust_feat)

        gates = self.compute_gates(patient_latent, context)

        gated_clinical = gates["clinical"] * enc_clinical
        gated_pastwork = gates["pastwork"] * enc_pastwork
        gated_logistics = gates["logistics"] * enc_logistics
        gated_trust = gates["trust"] * enc_trust

        gated_features = torch.cat(
            [gated_clinical, gated_pastwork, gated_logistics, gated_trust],
            dim=-1,
        )

        predictor_input = torch.cat([patient_latent, gated_features], dim=-1)
        predicted_ideal = self.predictor(predictor_input)
        return predicted_ideal, gates

    def forward(
        self,
        patient_emb: torch.Tensor,
        doctor_emb: torch.Tensor,
        clinical_feat: torch.Tensor,
        pastwork_feat: torch.Tensor,
        logistics_feat: torch.Tensor,
        trust_feat: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Full forward pass.

        Returns:
            dict with score, embeddings, gates, temperature
        """
        patient_latent = self.encode_patient(patient_emb)
        doctor_latent = self.encode_doctor(doctor_emb)

        predicted_ideal, gates = self.predict_ideal_doctor(
            patient_latent,
            clinical_feat,
            pastwork_feat,
            logistics_feat,
            trust_feat,
            context,
        )

        pred_proj = self.predictor_proj(predicted_ideal)
        doc_proj = self.doctor_proj(doctor_latent)

        pred_norm = F.normalize(pred_proj, dim=-1)
        doc_norm = F.normalize(doc_proj, dim=-1)
        score = (pred_norm * doc_norm).sum(dim=-1, keepdim=True)
        score = (score + 1.0) / 2.0

        gate_means = {
            "clinical": gates["clinical"].mean(dim=-1),
            "pastwork": gates["pastwork"].mean(dim=-1),
            "logistics": gates["logistics"].mean(dim=-1),
            "trust": gates["trust"].mean(dim=-1),
        }

        return {
            "score": score,
            "predicted_ideal": pred_proj,
            "doctor_embedding": doc_proj,
            "patient_latent": patient_latent,
            "doctor_latent": doctor_latent,
            "gates": gates,
            "gate_means": gate_means,
            "temperature": self.log_temperature.exp(),
        }

    def get_parameter_groups(self, base_lr: float) -> list[Dict[str, Any]]:
        """
        Get parameter groups with differential learning rates.
        Doctor encoder learns at 0.05× rate (VL-JEPA insight).
        """
        doctor_params = list(self.doctor_encoder.parameters()) + list(self.doctor_proj.parameters())
        other_params = [
            p
            for n, p in self.named_parameters()
            if "doctor_encoder" not in n and "doctor_proj" not in n
        ]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": doctor_params, "lr": base_lr * 0.05},
        ]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
