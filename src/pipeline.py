"""
Outline-driven demo pipeline.

This file is intentionally incomplete. Use Codex to implement it so that:
- `pytest -q` passes
- `python tools/validate_outline.py` passes
- behavior matches `outline.md`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any


def make_dataset(seed: int, n_samples: int) -> tuple[list[list[float]], list[int]]:
    """Generate a deterministic, linearly-separable-ish 2D dataset."""
    raise NotImplementedError("TODO: implement make_dataset")


def train_logreg(
    X: list[list[float]],
    y: list[int],
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    """Train logistic regression with batch gradient descent.

    Return a JSON-serializable dict with keys:
      - w: list[float]  # length 2
      - b: float
    """
    raise NotImplementedError("TODO: implement train_logreg")


def predict_proba(model: dict[str, Any], X: list[list[float]]) -> list[float]:
    """Return P(y=1|x) for each row in X."""
    raise NotImplementedError("TODO: implement predict_proba")


def evaluate(model: dict[str, Any], X: list[list[float]], y: list[int]) -> dict[str, float]:
    """Return metrics dict with numeric keys: accuracy, loss (log loss)."""
    raise NotImplementedError("TODO: implement evaluate")


def run(out_dir: str, seed: int, n_samples: int, epochs: int, lr: float) -> dict[str, Any]:
    """Run the full pipeline and write artifacts. Returns metrics dict."""
    raise NotImplementedError("TODO: implement run")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Outline-driven pipeline demo (stdlib only).")
    p.add_argument("--out", required=True, help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-samples", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    metrics = run(
        out_dir=args.out,
        seed=args.seed,
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
    )
    # required one-line summary
    print(f"accuracy={metrics['accuracy']:.6f} loss={metrics['loss']:.6f} out={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
