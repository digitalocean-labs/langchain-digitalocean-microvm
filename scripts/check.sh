#!/usr/bin/env bash
# The checks a pull request needs, in one command: lint, types, unit tests (local mode), the wording
# gate and a secret scan. Creates no MicroVMs and needs no credentials.
#
#   scripts/check.sh            # everything
#   scripts/check.sh --quick    # lint, types and the wording gate only (seconds)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

quick=0
[ "${1:-}" = "--quick" ] && quick=1

step() { printf '\n== %s\n' "$1"; }

step "ruff"
uv run ruff check .
step "pyright"
uv run pyright
step "wording gate"
scripts/grep_gate.sh
step "secret scan (full history)"
if command -v gitleaks >/dev/null 2>&1; then
    gitleaks git --log-opts="--all" --redact --no-banner .
else
    echo "gitleaks is not installed: skipped (CI pins the version in .github/workflows/ci.yml)"
fi
if [ "$quick" -eq 0 ]; then
    step "unit tests (local mode)"
    uv run pytest -q
fi
printf '\nall checks passed\n'
