# Codex Outline-Driven Coding Demo (local Codex + remote GPU via SSH)

This repo is intentionally **incomplete**. Your goal is to let **Codex** implement the pipeline from `outline.md`
and make all checks pass, with minimal human intervention.

## What you will run (one command)

From a Linux shell (WSL recommended on Windows):

```bash
bash tools/full_auto.sh --remote USER@YOUR_GPU_SERVER --repo-dir ~/repos/codex_outline_demo
```

What it does:
1) Runs Codex non-interactively to implement missing code (`src/pipeline.py`) to satisfy `outline.md` + tests.
2) Runs local checks (`pytest` + outline validation).
3) Commits & pushes to GitHub (you set origin).
4) SSH to the remote server, pulls the repo, runs the pipeline, and pulls back the run artifacts.

> If you don't want the remote part yet, run:
> `bash tools/local_auto.sh`

---

## Prereqs (local machine)

- Git
- Python 3.10+
- Node.js 18+
- Codex CLI: `npm i -g @openai/codex`
- Auth once: `codex login`  (or `codex login --api-key ...`)

On Windows, install and use **WSL2 Ubuntu** for the best Codex CLI experience.

---

## Prereqs (remote GPU server)

- git, python3
- Able to `git pull` from your GitHub repo (HTTPS or SSH)
- SSH access from your local machine

The demo pipeline has **no external Python dependencies** (stdlib only), so it runs anywhere.

---

## Project layout

- `outline.md`: human-readable programming outline (the "spec").
- `outline.yaml`: machine-checkable version of the spec (used by `tools/validate_outline.py`).
- `src/pipeline.py`: **TODO** implementation target (Codex should complete this).
- `tests/`: acceptance tests (Codex must make them pass).
- `tools/`: automation scripts for Codex + remote run.

---

## Manual run (after Codex finishes)

Local:
```bash
python -m src.pipeline --out runs/local_demo --seed 0 --n-samples 400 --epochs 100 --lr 0.2
python tools/validate_outline.py
pytest -q
```

Remote:
```bash
ssh USER@REMOTE "cd ~/repos/codex_outline_demo && python -m src.pipeline --out runs/remote_demo --seed 0 --n-samples 400 --epochs 100 --lr 0.2"
```
