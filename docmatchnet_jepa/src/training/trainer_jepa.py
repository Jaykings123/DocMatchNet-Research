"""Two-stage training workflow for JEPA experiments."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm

from evaluation.metrics import Evaluator
from models.losses import DocMatchJEPALoss


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


class JEPATrainer:
    """
    Two-stage training for DocMatchNet-JEPA.

    Stage 1: Alignment pretraining with InfoNCE
    Stage 2: Gated supervised fine-tuning
    """

    def __init__(self, model: torch.nn.Module, config: Any, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.loss_fn = DocMatchJEPALoss(config)

        stage2_lr = float(_cfg_get(config, "stage2", "learning_rate", default=1e-4))
        weight_decay = float(_cfg_get(config, "stage2", "weight_decay", default=1e-5))

        if hasattr(model, "get_parameter_groups"):
            param_groups = model.get_parameter_groups(stage2_lr)
        else:
            param_groups = [{"params": model.parameters(), "lr": stage2_lr}]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

        self.evaluator = Evaluator()
        self.history = {
            "train_loss": [],
            "val_metrics": [],
            "infonce_accuracy": [],
            "temperature": [],
        }

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train_stage1(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 30,
    ) -> Dict[str, list]:
        """
        Stage 1: Alignment pretraining.

        Focus on InfoNCE to align patient-doctor embedding space.
        Freeze gates at initial values.
        """
        del val_loader
        print("=" * 50)
        print("STAGE 1: Alignment Pretraining")
        print("=" * 50)

        for name, param in self.model.named_parameters():
            if "gate" in name:
                param.requires_grad = False

        best_val_loss = float("inf")
        _ = best_val_loss

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_temp = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Stage1 Epoch {epoch}"):
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()

                output = self.model(
                    batch["case_embedding"],
                    batch["positive_doctor_embedding"],
                    batch["positive_clinical"],
                    batch["positive_pastwork"],
                    batch["positive_logistics"],
                    batch["positive_trust"],
                    batch["context"],
                )

                loss, metrics = self.loss_fn(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += float(metrics["infonce_loss"])
                epoch_acc += float(metrics["infonce_accuracy"])
                epoch_temp += float(metrics["temperature"])
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_acc = epoch_acc / max(n_batches, 1)
            avg_temp = epoch_temp / max(n_batches, 1)

            print(f"Epoch {epoch}: InfoNCE Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

            self.history["train_loss"].append(avg_loss)
            self.history["infonce_accuracy"].append(avg_acc)
            self.history["temperature"].append(avg_temp)

            self.scheduler.step()

        for _, param in self.model.named_parameters():
            param.requires_grad = True

        return self.history

    def train_stage2(
        self,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        epochs: int = 50,
        patience: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, list]]:
        """
        Stage 2: Gated supervised fine-tuning.

        Full training with gates active.
        """
        print("=" * 50)
        print("STAGE 2: Gated SFT")
        print("=" * 50)

        best_ndcg = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Stage2 Epoch {epoch}"):
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()

                output = self.model(
                    batch["case_embedding"],
                    batch["positive_doctor_embedding"],
                    batch["positive_clinical"],
                    batch["positive_pastwork"],
                    batch["positive_logistics"],
                    batch["positive_trust"],
                    batch["context"],
                )

                loss, metrics = self.loss_fn(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += float(metrics["total_loss"])
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            val_results = self.evaluator.evaluate(self.model, val_loader, self.device)
            val_ndcg = float(val_results["overall"]["ndcg@5"][0])

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val NDCG@5 = {val_ndcg:.4f}")

            self.history["train_loss"].append(avg_loss)
            self.history["val_metrics"].append(val_results["overall"])

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            self.scheduler.step()

        self.model.load_state_dict(torch.load("best_model.pt", map_location=self.device))
        test_results = self.evaluator.evaluate(self.model, test_loader, self.device)
        return test_results, self.history

    def train_full(
        self,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, list]]:
        """Run both training stages."""
        stage1_epochs = int(_cfg_get(self.config, "stage1", "epochs", default=30))
        stage2_epochs = int(_cfg_get(self.config, "stage2", "epochs", default=50))
        stage2_patience = int(_cfg_get(self.config, "stage2", "patience", default=10))

        self.train_stage1(train_loader, val_loader, epochs=stage1_epochs)
        test_results, history = self.train_stage2(
            train_loader,
            val_loader,
            test_loader,
            epochs=stage2_epochs,
            patience=stage2_patience,
        )
        return test_results, history
