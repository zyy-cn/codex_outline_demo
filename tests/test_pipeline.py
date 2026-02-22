import json
import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_creates_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "run1"
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline",
        "--out",
        str(out),
        "--seed",
        "0",
        "--n-samples",
        "200",
        "--epochs",
        "50",
        "--lr",
        "0.2",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "accuracy=" in r.stdout and "loss=" in r.stdout and "out=" in r.stdout

    for name in ["config.json", "model.json", "metrics.json"]:
        p = out / name
        assert p.exists(), f"missing {name}"

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics and "loss" in metrics
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0
    assert float(metrics["loss"]) >= 0.0


def test_determinism(tmp_path: Path) -> None:
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    base = [
        sys.executable,
        "-m",
        "src.pipeline",
        "--seed",
        "123",
        "--n-samples",
        "300",
        "--epochs",
        "60",
        "--lr",
        "0.15",
    ]
    r1 = subprocess.run(base + ["--out", str(out1)], capture_output=True, text=True)
    r2 = subprocess.run(base + ["--out", str(out2)], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr

    m1 = json.loads((out1 / "metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "metrics.json").read_text(encoding="utf-8"))
    assert m1 == m2, "metrics must be identical for same seed/args"
