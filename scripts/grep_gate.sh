#!/usr/bin/env bash
# Public-wording gate: everything in this repository must read as public-ready.
# Fails on dates, release-stage wording, planning labels and internal references in tracked
# (and untracked, not ignored) files. Exempt: lock files, recorded API fixtures, ISO timestamps
# with a time part (synthetic test data).
#
#   scripts/grep_gate.sh          # exit 1 and print the hits, if any
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pattern='20[0-9]{2}-[0-9]{2}-[0-9]{2}([^T0-9]|$)'        # dates (of runs, plans, releases)
pattern+='|[Pp]review|[Pp]rivate [Bb]eta|\bGA\b'         # release-stage wording
pattern+='|[Pp]hase ?[0-9]|\bM[1-9]\b|\bP[1-6]\b|ADR-'   # planning labels
pattern+='|[Pp]rod(uct)? [Tt]eam|PM to confirm|hrprobe|internal-repo|github\.internal|CLAUDE\.md|harness-runtime-evaluation'  # internal references

hits=$(git grep --untracked -nIE "$pattern" -- . \
    ':!*uv.lock' \
    ':!scripts/grep_gate.sh' \
    ':!tests/unit_tests/fixtures' || true)
if [ -n "$hits" ]; then
    echo "$hits"
    echo "grep gate: the hits above are not public-ready" >&2
    exit 1
fi
echo "grep gate: clean"
