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
from typing import Any


def make_dataset(seed: int, n_samples: int) -> tuple[list[list[float]], list[int]]:
    """Generate a deterministic, linearly-separable-ish 2D dataset."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []

    for _ in range(n_samples):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        noise = rng.uniform(-0.35, 0.35)
        score = (1.4 * x1) + (0.9 * x2) - 0.1 + noise
        label = 1 if score >= 0.0 else 0
        X.append([x1, x2])
        y.append(label)

    return X, y


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
    if not X or not y:
        raise ValueError("X and y must be non-empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if epochs < 0:
        raise ValueError("epochs must be >= 0")

    n_features = len(X[0])
    if n_features == 0:
        raise ValueError("X must contain at least one feature")
    if any(len(row) != n_features for row in X):
        raise ValueError("all rows in X must have the same length")

    w = [0.0 for _ in range(n_features)]
    b = 0.0
    n = float(len(X))

    for _ in range(epochs):
        grad_w = [0.0 for _ in range(n_features)]
        grad_b = 0.0

        for row, target in zip(X, y):
            z = b + sum(weight * value for weight, value in zip(w, row))
            p = _sigmoid(z)
            err = p - float(target)
            for j, value in enumerate(row):
                grad_w[j] += err * value
            grad_b += err

        for j in range(n_features):
            w[j] -= lr * (grad_w[j] / n)
        b -= lr * (grad_b / n)

    return {"w": w, "b": b}


def predict_proba(model: dict[str, Any], X: list[list[float]]) -> list[float]:
    """Return P(y=1|x) for each row in X."""
    weights_raw = model.get("w")
    bias_raw = model.get("b")
    if not isinstance(weights_raw, list):
        raise ValueError("model['w'] must be a list")
    if not isinstance(bias_raw, (int, float)):
        raise ValueError("model['b'] must be numeric")

    w = [float(v) for v in weights_raw]
    b = float(bias_raw)
    n_features = len(w)

    probs: list[float] = []
    for row in X:
        if len(row) != n_features:
            raise ValueError("row feature count does not match model weights")
        z = b + sum(weight * float(value) for weight, value in zip(w, row))
        probs.append(_sigmoid(z))
    return probs


def evaluate(model: dict[str, Any], X: list[list[float]], y: list[int]) -> dict[str, float]:
    """Return metrics dict with numeric keys: accuracy, loss (log loss)."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if not X:
        raise ValueError("X and y must be non-empty")

    probs = predict_proba(model, X)
    correct = 0
    loss_sum = 0.0
    eps = 1e-12

    for p, target in zip(probs, y):
        pred = 1 if p >= 0.5 else 0
        if pred == int(target):
            correct += 1

        p_clip = min(max(p, eps), 1.0 - eps)
        t = float(target)
        loss_sum += -(t * math.log(p_clip) + (1.0 - t) * math.log(1.0 - p_clip))

    n = float(len(y))
    return {"accuracy": correct / n, "loss": loss_sum / n}


def run(out_dir: str, seed: int, n_samples: int, epochs: int, lr: float) -> dict[str, Any]:
    """Run the full pipeline and write artifacts. Returns metrics dict."""
    X, y = make_dataset(seed=seed, n_samples=n_samples)
    model = train_logreg(X=X, y=y, epochs=epochs, lr=lr)
    metrics = evaluate(model=model, X=X, y=y)

    os.makedirs(out_dir, exist_ok=True)

    config = {
        "seed": seed,
        "n_samples": n_samples,
        "epochs": epochs,
        "lr": lr,
    }

    _write_json(os.path.join(out_dir, "config.json"), config)
    _write_json(os.path.join(out_dir, "model.json"), model)
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)

    return metrics


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write("\n")


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
