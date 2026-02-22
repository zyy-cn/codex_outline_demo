#!/usr/bin/env bash
set -euo pipefail

# Local-only automation: let Codex implement missing code and make tests pass.
# Usage: bash tools/local_auto.sh

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found. Install: npm i -g @openai/codex"
  exit 1
fi

PROMPT=$(cat <<'EOF'
You are working in a small Python repo.
Goal: implement src/pipeline.py to satisfy outline.md, outline.yaml, and tests/.

Constraints:
- stdlib only (no external deps).
- Deterministic metrics.json for same seed/args.
- Must write config.json, model.json, metrics.json under --out.
- Keep code clean, typed, and readable.

Process:
1) Read outline.md and outline.yaml.
2) Implement missing functions in src/pipeline.py.
3) Run: python tools/validate_outline.py
4) Run: pytest -q
5) If failing, iterate until both pass.
Stop when everything passes.
EOF
)

# Full auto, sandboxed to workspace writes.
codex exec --full-auto --sandbox workspace-write "$PROMPT"

echo
echo "Local automation finished. Run checks:"
echo "  python tools/validate_outline.py"
echo "  pytest -q"
