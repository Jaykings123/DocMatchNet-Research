"""Dataset classes and dataloader helpers for DocMatchNet experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return torch.load(path, map_location="cpu")


def _load_optional(path: Path, default: Any = None) -> Any:
    return torch.load(path, map_location="cpu") if path.exists() else default


class _BaseDocMatchDataset(Dataset):
    """Shared loading and split logic for all dataset variants."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.split = split

        self.case_embeddings = _load_tensor(self.data_dir / "case_embeddings.pt").float()
        self.doctor_embeddings = _load_tensor(self.data_dir / "doctor_embeddings.pt").float()
        self.clinical_features = _load_tensor(self.data_dir / "clinical_features.pt").float()
        self.pastwork_features = _load_tensor(self.data_dir / "pastwork_features.pt").float()
        self.logistics_features = _load_tensor(self.data_dir / "logistics_features.pt").float()
        self.trust_features = _load_tensor(self.data_dir / "trust_features.pt").float()
        self.context_features = _load_tensor(self.data_dir / "context_features.pt").float()

        self.relevance_labels = _load_optional(self.data_dir / "relevance_labels.pt")
        if self.relevance_labels is None:
            self.relevance_labels = _load_tensor(self.data_dir / "relevance_matrix.pt")
        self.relevance_labels = self.relevance_labels.long()

        self.doctor_indices = _load_optional(self.data_dir / "doctor_indices.pt")
        if self.doctor_indices is None:
            n_cases = self.case_embeddings.shape[0]
            n_cand = self.relevance_labels.shape[1]
            self.doctor_indices = torch.arange(n_cand).unsqueeze(0).repeat(n_cases, 1)
        self.doctor_indices = self.doctor_indices.long()

        self.context_categories = self._load_context_categories()
        self.indices = self._get_split_indices(split)

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def _load_context_categories(self) -> np.ndarray:
        categories = _load_optional(self.data_dir / "context_categories.pt")
        if categories is not None:
            if isinstance(categories, torch.Tensor):
                categories = categories.cpu().numpy()
            return np.asarray(categories).astype(str)

        # Infer category from context feature channels:
        # [urgency, symptom_count, red_flag, age, comorbidity, rarity, is_pediatric, is_emergency]
        ctx = self.context_features.numpy()
        is_pediatric = ctx[:, 6] > 0.5
        is_emergency = ctx[:, 7] > 0.5
        rarity = ctx[:, 5]
        comorbidity = ctx[:, 4]

        out = np.full(ctx.shape[0], "routine", dtype=object)
        out[(comorbidity > 0.4) & ~is_emergency & ~is_pediatric] = "complex"
        out[(rarity > 0.65) & ~is_emergency & ~is_pediatric] = "rare_disease"
        out[is_emergency] = "emergency"
        out[is_pediatric] = "pediatric"
        return out.astype(str)

    def _get_split_indices(self, split: str) -> np.ndarray:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        train_ratio = float(self._cfg("data", "train_ratio", default=0.7))
        val_ratio = float(self._cfg("data", "val_ratio", default=0.15))
        test_ratio = float(self._cfg("data", "test_ratio", default=0.15))
        total = train_ratio + val_ratio + test_ratio
        train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

        all_indices = np.arange(len(self.case_embeddings))
        split_map: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        categories = self.context_categories

        for category in np.unique(categories):
            cat_idx = all_indices[categories == category].copy()
            self.rng.shuffle(cat_idx)

            n = len(cat_idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val

            split_map["train"].extend(cat_idx[:n_train].tolist())
            split_map["val"].extend(cat_idx[n_train : n_train + n_val].tolist())
            split_map["test"].extend(cat_idx[n_train + n_val : n_train + n_val + n_test].tolist())

        out = np.asarray(split_map[split], dtype=np.int64)
        self.rng.shuffle(out)
        return out

    def __len__(self) -> int:
        return int(len(self.indices))


class DocMatchDatasetJEPA(_BaseDocMatchDataset):
    """
    Dataset for JEPA-style training with InfoNCE loss.

    Returns batches where each sample represents a case,
    with positive and negative doctors for contrastive learning.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        negatives_per_positive: int = 1,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        """
        Load pre-computed features from .pt files.

        Args:
            data_dir: path to saved .pt files
            split: 'train', 'val', or 'test'
            negatives_per_positive: for hard negative mining
        """
        super().__init__(data_dir=data_dir, split=split, config=config, seed=seed)
        self.negatives_per_positive = max(1, int(negatives_per_positive))

    def _choose_positive_negative_positions(self, relevance: np.ndarray) -> Tuple[int, np.ndarray]:
        pos_pool = np.where(relevance >= 3)[0]
        if len(pos_pool) == 0:
            pos_pool = np.where(relevance == relevance.max())[0]
        pos_pos = int(self.rng.choice(pos_pool))

        easy_neg_pool = np.where(relevance <= 1)[0]
        hard_neg_pool = np.where(relevance == 2)[0]

        neg_positions: List[int] = []
        for _ in range(self.negatives_per_positive):
            use_hard = (self.rng.random() < 0.2) and (len(hard_neg_pool) > 0)
            if use_hard:
                neg_positions.append(int(self.rng.choice(hard_neg_pool)))
            elif len(easy_neg_pool) > 0:
                neg_positions.append(int(self.rng.choice(easy_neg_pool)))
            elif len(hard_neg_pool) > 0:
                neg_positions.append(int(self.rng.choice(hard_neg_pool)))
            else:
                fallback = np.where(np.arange(len(relevance)) != pos_pos)[0]
                neg_positions.append(int(self.rng.choice(fallback)))
        return pos_pos, np.asarray(neg_positions, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict for contrastive learning.
        """
        case_idx = int(self.indices[idx])
        relevance = self.relevance_labels[case_idx].cpu().numpy()
        cand_doc_indices = self.doctor_indices[case_idx]

        pos_pos, neg_pos = self._choose_positive_negative_positions(relevance)
        pos_doc_idx = int(cand_doc_indices[pos_pos].item())
        neg_doc_idx = cand_doc_indices[torch.from_numpy(neg_pos)]

        return {
            "case_embedding": self.case_embeddings[case_idx],  # (384,)
            "positive_doctor_embedding": self.doctor_embeddings[pos_doc_idx],  # (384,)
            "negative_doctor_embeddings": self.doctor_embeddings[neg_doc_idx],  # (n_neg, 384)
            "positive_clinical": self.clinical_features[case_idx, pos_pos],  # (4,)
            "positive_pastwork": self.pastwork_features[case_idx, pos_pos],  # (5,)
            "positive_logistics": self.logistics_features[case_idx, pos_pos],  # (5,)
            "positive_trust": self.trust_features[case_idx, pos_pos],  # (3,)
            "negative_clinical": self.clinical_features[case_idx, neg_pos],  # (n_neg, 4)
            "negative_pastwork": self.pastwork_features[case_idx, neg_pos],  # (n_neg, 5)
            "negative_logistics": self.logistics_features[case_idx, neg_pos],  # (n_neg, 5)
            "negative_trust": self.trust_features[case_idx, neg_pos],  # (n_neg, 3)
            "context": self.context_features[case_idx],  # (8,)
            "positive_relevance": int(relevance[pos_pos]),  # 3 or 4
            "negative_relevances": torch.from_numpy(relevance[neg_pos]).long(),  # (n_neg,) 0/1 (or 2 for hard)
        }


class DocMatchDatasetOriginal(_BaseDocMatchDataset):
    """
    Dataset for original DocMatchNet (pairwise ranking).

    Returns positive/negative pairs for margin ranking loss.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, config=config, seed=seed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict for pairwise training.
        """
        case_idx = int(self.indices[idx])
        relevance = self.relevance_labels[case_idx].cpu().numpy()
        cand_doc_indices = self.doctor_indices[case_idx]

        pos_pool = np.where(relevance >= 3)[0]
        if len(pos_pool) == 0:
            pos_pool = np.where(relevance == relevance.max())[0]
        neg_pool = np.where(relevance <= 1)[0]
        if len(neg_pool) == 0:
            neg_pool = np.where(relevance == relevance.min())[0]

        pos_pos = int(self.rng.choice(pos_pool))
        neg_pos = int(self.rng.choice(neg_pool))
        pos_doc_idx = int(cand_doc_indices[pos_pos].item())
        neg_doc_idx = int(cand_doc_indices[neg_pos].item())

        return {
            "case_embedding": self.case_embeddings[case_idx],  # (384,)
            "pos_doctor_embedding": self.doctor_embeddings[pos_doc_idx],  # (384,)
            "neg_doctor_embedding": self.doctor_embeddings[neg_doc_idx],  # (384,)
            "pos_features": {
                "clinical": self.clinical_features[case_idx, pos_pos],
                "pastwork": self.pastwork_features[case_idx, pos_pos],
                "logistics": self.logistics_features[case_idx, pos_pos],
                "trust": self.trust_features[case_idx, pos_pos],
            },
            "neg_features": {
                "clinical": self.clinical_features[case_idx, neg_pos],
                "pastwork": self.pastwork_features[case_idx, neg_pos],
                "logistics": self.logistics_features[case_idx, neg_pos],
                "trust": self.trust_features[case_idx, neg_pos],
            },
            "context": self.context_features[case_idx],  # (8,)
            "pos_label": torch.tensor(1.0, dtype=torch.float32),
            "neg_label": torch.tensor(0.0, dtype=torch.float32),
        }


class DocMatchDatasetEval(_BaseDocMatchDataset):
    """
    Dataset for evaluation (listwise).

    Returns all doctors for a case to compute ranking metrics.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, config=config, seed=seed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns dict with all selected doctors for this case.
        """
        case_idx = int(self.indices[idx])
        cand_doc_indices = self.doctor_indices[case_idx]
        return {
            "case_embedding": self.case_embeddings[case_idx],  # (384,)
            "doctor_embeddings": self.doctor_embeddings[cand_doc_indices],  # (100, 384)
            "clinical_features": self.clinical_features[case_idx],  # (100, 4)
            "pastwork_features": self.pastwork_features[case_idx],  # (100, 5)
            "logistics_features": self.logistics_features[case_idx],  # (100, 5)
            "trust_features": self.trust_features[case_idx],  # (100, 3)
            "context": self.context_features[case_idx],  # (8,)
            "relevance_labels": self.relevance_labels[case_idx],  # (100,)
            "context_category": str(self.context_categories[case_idx]),
        }


def create_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    mode: str = "jepa",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        mode: 'jepa' for JEPA training, 'original' for DocMatchNet-Original

    Returns:
        train_loader, val_loader, test_loader
    """
    if mode not in {"jepa", "original"}:
        raise ValueError(f"mode must be one of ['jepa', 'original'], got: {mode}")

    num_workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", True))
    seed = int(config.get("seed", 42))
    negatives_per_positive = int(config.get("negatives_per_positive", 1))

    if mode == "jepa":
        train_bs = int(config.get("stage1", {}).get("batch_size", 256))
        eval_bs = int(config.get("stage2", {}).get("batch_size", 128))
        train_ds = DocMatchDatasetJEPA(
            data_dir=data_dir,
            split="train",
            negatives_per_positive=negatives_per_positive,
            config=config,
            seed=seed,
        )
        val_ds = DocMatchDatasetJEPA(
            data_dir=data_dir,
            split="val",
            negatives_per_positive=negatives_per_positive,
            config=config,
            seed=seed + 1,
        )
    else:
        train_bs = int(config.get("stage2", {}).get("batch_size", 128))
        eval_bs = train_bs
        train_ds = DocMatchDatasetOriginal(
            data_dir=data_dir,
            split="train",
            config=config,
            seed=seed,
        )
        val_ds = DocMatchDatasetOriginal(
            data_dir=data_dir,
            split="val",
            config=config,
            seed=seed + 1,
        )

    test_ds = DocMatchDatasetEval(
        data_dir=data_dir,
        split="test",
        config=config,
        seed=seed + 2,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
