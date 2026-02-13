"""Utilities for running multi-seed experiments."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def run_multi_seed_experiment(
    model_class: type,
    model_kwargs: Dict[str, Any],
    trainer_class: Any,
    trainer_kwargs: Dict[str, Any],
    train_loader_fn: Callable[[int], Any],
    val_loader: Any,
    test_loader: Any,
    seeds: Iterable[int] = (42, 123, 456),
    experiment_name: str = "default",
) -> Dict[str, Any]:
    """
    Run experiment with multiple seeds and aggregate results.

    Args:
        model_class: class to instantiate
        model_kwargs: dict of kwargs for model
        trainer_class: training function or class
        trainer_kwargs: dict of kwargs for trainer
        train_loader_fn: function that returns train_loader (takes seed)
        val_loader: validation dataloader
        test_loader: test dataloader
        seeds: list/iterable of random seeds
        experiment_name: name for saving results

    Returns:
        aggregated_results: dict with mean, std, per-run values
    """
    all_run_results: List[Dict[str, Any]] = []
    seed_list = list(seeds)

    # Determine device: prefer explicit trainer_kwargs, else auto
    device = trainer_kwargs.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    for seed_idx, seed in enumerate(seed_list):
        print(f"\n{'=' * 60}")
        print(f"RUN {seed_idx + 1}/{len(seed_list)} | Seed: {seed}")
        print(f"{'=' * 60}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create model
        model = model_class(**model_kwargs).to(device)

        # Create train loader with this seed's shuffling
        train_loader = train_loader_fn(seed)

        # Train
        if hasattr(trainer_class, "train_full"):
            # JEPA two-stage training (trainer_class is a class)
            trainer = trainer_class(model, device=device, **trainer_kwargs)
            test_results, _history = trainer.train_full(
                train_loader, val_loader, test_loader
            )
        else:
            # Single-stage training (trainer_class is a function)
            test_results = trainer_class(
                model, train_loader, val_loader, test_loader, device=device, **trainer_kwargs
            )

        all_run_results.append(test_results)

        # Save individual run
        torch.save(
            model.state_dict(),
            f"/kaggle/working/results/{experiment_name}_seed{seed}.pt",
        )

    # Aggregate results
    aggregated: Dict[str, Any] = {}
    metrics = list(all_run_results[0]["overall"].keys())

    for metric in metrics:
        values = [r["overall"][metric][0] for r in all_run_results]
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "runs": [float(v) for v in values],
            "ci_95_lower": float(np.percentile(values, 2.5)),
            "ci_95_upper": float(np.percentile(values, 97.5)),
        }

    # Aggregate stratified results
    aggregated["stratified"] = {}
    contexts = list(all_run_results[0].get("stratified", {}).keys())
    for ctx in contexts:
        ctx_values = [
            r["stratified"][ctx][0]
            for r in all_run_results
            if ctx in r.get("stratified", {})
        ]
        if ctx_values:
            aggregated["stratified"][ctx] = {
                "mean": float(np.mean(ctx_values)),
                "std": float(np.std(ctx_values)),
            }

    # Save aggregated results
    with open(f"/kaggle/working/results/{experiment_name}_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATED RESULTS ({len(seed_list)} seeds)")
    print(f"{'=' * 60}")
    for metric, vals in aggregated.items():
        if isinstance(vals, dict) and "mean" in vals:
            print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    return aggregated
