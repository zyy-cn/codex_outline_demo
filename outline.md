# Programming Outline (Demo)

## Goal
Implement a tiny, deterministic ML-style pipeline **from scratch** (stdlib only) that:
1) generates a synthetic binary classification dataset,
2) trains a logistic regression model with gradient descent,
3) evaluates accuracy,
4) saves artifacts to an output folder.

This is a stand-in for a paper "pipeline": outline → implementation → verification.

## CLI contract
Provide a CLI entrypoint:

```bash
python -m src.pipeline --out <OUT_DIR> --seed <INT> --n-samples <INT> --epochs <INT> --lr <FLOAT>
```

## Functional requirements
- Implement these functions in `src/pipeline.py`:
  - `make_dataset(seed: int, n_samples: int) -> tuple[list[list[float]], list[int]]`
  - `train_logreg(X, y, epochs: int, lr: float) -> dict` (returns a serializable model dict)
  - `predict_proba(model: dict, X) -> list[float]`
  - `evaluate(model: dict, X, y) -> dict` (must include `accuracy` and `loss`)
  - `run(out_dir: str, seed: int, n_samples: int, epochs: int, lr: float) -> dict`

- Determinism:
  - Same `seed` + args ⇒ identical `metrics.json` contents.

- Artifacts:
  - Write these files under `--out`:
    - `config.json` (the resolved CLI args)
    - `model.json` (trained weights/bias)
    - `metrics.json` (must contain numeric `accuracy` and `loss`)
  - Print a one-line summary to stdout: `accuracy=<...> loss=<...> out=<...>`

## Validation requirements
- `pytest -q` must pass.
- `python tools/validate_outline.py` must pass.

## Non-goals
- High accuracy. This is a workflow demo, not a benchmark.
- External dependencies (numpy/torch/etc). Keep it stdlib only.
