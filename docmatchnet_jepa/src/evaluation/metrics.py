"""Evaluation metrics and statistical tests for DocMatchNet."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy import stats


class RankingMetrics:
    """Compute ranking quality metrics."""

    @staticmethod
    def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        Args:
            scores: (n_doctors,) predicted scores
            labels: (n_doctors,) relevance labels (0-4)
            k: cutoff position

        Returns:
            float: NDCG@K in [0, 1]
        """
        ranking = np.argsort(-scores)[:k]
        dcg = 0.0
        for i, doc_idx in enumerate(ranking):
            rel = labels[doc_idx]
            dcg += (2**rel - 1) / np.log2(i + 2)

        ideal_ranking = np.argsort(-labels)[:k]
        idcg = 0.0
        for i, doc_idx in enumerate(ideal_ranking):
            rel = labels[doc_idx]
            idcg += (2**rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 0.0
        return float(dcg / idcg)

    @staticmethod
    def map_score(scores: np.ndarray, labels: np.ndarray, threshold: int = 2) -> float:
        """
        Mean Average Precision.
        Relevant = labels >= threshold.
        """
        ranking = np.argsort(-scores)
        relevant = labels >= threshold

        precisions: List[float] = []
        relevant_count = 0
        for i, doc_idx in enumerate(ranking):
            if relevant[doc_idx]:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        if len(precisions) == 0:
            return 0.0
        return float(np.mean(precisions))

    @staticmethod
    def mrr(scores: np.ndarray, labels: np.ndarray, threshold: int = 2) -> float:
        """Mean Reciprocal Rank."""
        ranking = np.argsort(-scores)
        relevant = labels >= threshold
        for i, doc_idx in enumerate(ranking):
            if relevant[doc_idx]:
                return float(1.0 / (i + 1))
        return 0.0

    @staticmethod
    def hit_rate_at_k(scores: np.ndarray, labels: np.ndarray, k: int, threshold: int = 2) -> float:
        """Whether any relevant item is in top K."""
        top_k = np.argsort(-scores)[:k]
        relevant = labels >= threshold
        return float(relevant[top_k].any())

    @staticmethod
    def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int, threshold: int = 2) -> float:
        """Precision at K."""
        top_k = np.argsort(-scores)[:k]
        relevant = labels >= threshold
        return float(relevant[top_k].sum() / k)


class Evaluator:
    """Evaluate model on full test set."""

    def __init__(self):
        self.metrics = RankingMetrics()

    def evaluate(self, model: Any, dataloader: Iterable, device: str = "cuda") -> Dict[str, Any]:
        """
        Evaluate model on dataloader.

        Returns:
            results: dict of metric_name -> (mean, std)
            per_case: dict of metric_name -> list of values
            per_context: dict of context -> dict of metrics
        """
        model.eval()

        all_results = {
            "ndcg@1": [],
            "ndcg@5": [],
            "ndcg@10": [],
            "map": [],
            "mrr": [],
            "hr@5": [],
            "hr@10": [],
            "p@3": [],
            "p@5": [],
        }

        context_results = {
            "routine": {k: [] for k in all_results},
            "complex": {k: [] for k in all_results},
            "rare_disease": {k: [] for k in all_results},
            "emergency": {k: [] for k in all_results},
            "pediatric": {k: [] for k in all_results},
        }

        gate_activations = {"clinical": [], "pastwork": [], "logistics": [], "trust": []}
        context_gates = {ctx: {g: [] for g in gate_activations} for ctx in context_results}

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                num_docs = int(batch["doctor_embeddings"].shape[1])
                scores_list = []
                gates_list = []

                for i in range(num_docs):
                    output = model(
                        batch["case_embedding"],
                        batch["doctor_embeddings"][:, i],
                        batch["clinical_features"][:, i],
                        batch["pastwork_features"][:, i],
                        batch["logistics_features"][:, i],
                        batch["trust_features"][:, i],
                        batch["context"],
                    )
                    scores_list.append(output["score"])
                    if "gate_means" in output:
                        gates_list.append(output["gate_means"])

                scores = torch.cat(scores_list, dim=-1).detach().cpu().numpy()
                labels = batch["relevance_labels"].detach().cpu().numpy()
                contexts = batch["context_category"]
                if isinstance(contexts, tuple):
                    contexts = list(contexts)

                for b in range(scores.shape[0]):
                    s = scores[b]
                    l = labels[b]
                    ctx = contexts[b]
                    if ctx not in context_results:
                        ctx = "routine"

                    case_metrics = {
                        "ndcg@1": self.metrics.ndcg_at_k(s, l, 1),
                        "ndcg@5": self.metrics.ndcg_at_k(s, l, 5),
                        "ndcg@10": self.metrics.ndcg_at_k(s, l, 10),
                        "map": self.metrics.map_score(s, l),
                        "mrr": self.metrics.mrr(s, l),
                        "hr@5": self.metrics.hit_rate_at_k(s, l, 5),
                        "hr@10": self.metrics.hit_rate_at_k(s, l, 10),
                        "p@3": self.metrics.precision_at_k(s, l, 3),
                        "p@5": self.metrics.precision_at_k(s, l, 5),
                    }

                    for metric, value in case_metrics.items():
                        all_results[metric].append(value)
                        context_results[ctx][metric].append(value)

                if gates_list:
                    # Aggregate gate means over doctor loop for each case in batch.
                    gate_names = gate_activations.keys()
                    for gate_name in gate_names:
                        stacked = torch.stack([g[gate_name] for g in gates_list], dim=0).mean(dim=0)
                        gate_vals = stacked.detach().cpu().numpy()
                        gate_activations[gate_name].extend(gate_vals.tolist())
                        for b, ctx in enumerate(contexts):
                            if ctx in context_gates:
                                context_gates[ctx][gate_name].append(float(gate_vals[b]))

        final_results = {k: (float(np.mean(v)), float(np.std(v))) for k, v in all_results.items() if len(v) > 0}

        stratified_results: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for ctx, metrics in context_results.items():
            stratified_results[ctx] = {
                k: (float(np.mean(v)), float(np.std(v)))
                for k, v in metrics.items()
                if len(v) > 0
            }

        gate_stats = {}
        for gate, values in gate_activations.items():
            if values:
                gate_stats[gate] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        context_gate_stats = {}
        for ctx, gates in context_gates.items():
            context_gate_stats[ctx] = {
                gate: float(np.mean(values)) if values else 0.0
                for gate, values in gates.items()
            }

        return {
            "overall": final_results,
            "stratified": stratified_results,
            "per_case": all_results,
            "gate_stats": gate_stats,
            "context_gate_stats": context_gate_stats,
        }


class SignificanceTests:
    """Statistical significance testing."""

    @staticmethod
    def paired_t_test(values_a: Iterable[float], values_b: Iterable[float]) -> Tuple[float, float]:
        """Paired t-test between two methods."""
        a = np.asarray(list(values_a), dtype=float)
        b = np.asarray(list(values_b), dtype=float)
        t_stat, p_value = stats.ttest_rel(a, b)
        return float(t_stat), float(p_value)

    @staticmethod
    def wilcoxon_test(values_a: Iterable[float], values_b: Iterable[float]) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (non-parametric)."""
        a = np.asarray(list(values_a), dtype=float)
        b = np.asarray(list(values_b), dtype=float)
        stat, p_value = stats.wilcoxon(a, b)
        return float(stat), float(p_value)

    @staticmethod
    def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        arr = np.asarray(list(values), dtype=float)
        n = len(arr)
        if n == 0:
            return 0.0, 0.0

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=n, replace=True)
            bootstrap_means.append(float(np.mean(sample)))

        lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
        return float(lower), float(upper)

    @staticmethod
    def bonferroni_correction(p_values: Iterable[float], alpha: float = 0.05) -> Tuple[float, List[bool]]:
        """Apply Bonferroni correction."""
        p_vals = list(p_values)
        n_tests = max(len(p_vals), 1)
        corrected_alpha = alpha / n_tests
        significant = [p < corrected_alpha for p in p_vals]
        return float(corrected_alpha), significant
