#!/usr/bin/env bash
# Build the demo repository: two commits, the second is the bad deploy. Push it to a repository of your own.
#
#   service/make_demo_repo.sh /tmp/checkout-demo
#   cd /tmp/checkout-demo && git remote add origin https://github.com/<you>/<repo>.git && git push --force origin HEAD:main
#
# Commit 1 is the good service (payment confirmation through the provider's webhook). Commit 2 switches it to
# waiting for the provider's answer inside the request, which exhausts the worker pool.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${1:?usage: make_demo_repo.sh <empty directory>}"
mkdir -p "$dest/service" && cd "$dest"
git init -q -b main
cp "$here/app.py" service/app.py
cat > README.md <<'README'
# checkout service

A small, stateless service: `service/app.py`. Orders live in an external database.
README
# the first commit is two hours older, so the history reads as one
early="$(date -u -v-2H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '2 hours ago' +%Y-%m-%dT%H:%M:%SZ)"
git add -A && GIT_AUTHOR_DATE="$early" GIT_COMMITTER_DATE="$early" git -c user.name=demo -c user.email=demo@example.com commit -q -m "checkout: confirm payments through the provider's webhook"
sed -i.bak 's|^CONFIRMATION_MODE = "async".*|CONFIRMATION_MODE = "sync"  # wait for the provider answer inside the request (about 3.4 s)|' service/app.py && rm service/app.py.bak
git add -A && git -c user.name=demo -c user.email=demo@example.com commit -q -m "checkout: wait for the payment provider's answer inside the request"
git log --oneline
