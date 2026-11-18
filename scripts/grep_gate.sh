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
pattern+='|PM to confirm'                                # unresolved notes
# Extra patterns, one regular expression per line, from an untracked file (GREP_GATE_EXTRA, default
# .grep_gate_extra in the repository root).
extra="${GREP_GATE_EXTRA:-.grep_gate_extra}"
if [ -f "$extra" ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && pattern+="|$line"
    done < "$extra"
fi

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
