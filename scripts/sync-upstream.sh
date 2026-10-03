#!/usr/bin/env bash
#
# Rebase this fork onto the newest upstream/master.
#
# The fork is a patch series applied on top of upstream: upstream/master is the
# base, the fork's commits go on top. This fetches upstream, prints the upstream
# commits that overlap the fork (so nothing is missed), then rebases the given
# branch onto upstream/master. Conflicts are resolved by hand; rerere replays
# earlier resolutions automatically.
#
# usage: scripts/sync-upstream.sh [branch]     (default: the current branch)
#
set -euo pipefail

BRANCH="${1:-$(git branch --show-current)}"
if [ -z "$BRANCH" ]; then
    echo "no branch given and HEAD is detached" >&2
    exit 1
fi

echo "== fetch upstream =="
git fetch upstream

BASE=$(git merge-base "$BRANCH" upstream/master)
echo
echo "== upstream commits since the fork base ($BASE) =="
echo "   (keyword filter: things that overlap this fork)"
git log --oneline "$BASE"..upstream/master |
    grep -iE 'qwen4exp|qsa|kpool|mtp|nextn|decision|systemone|hip|rocm|cuda|strix|flash|sparse|top.?k' || echo "   (none matched)"

echo
echo "== rebase $BRANCH onto upstream/master =="
git switch "$BRANCH"
git rebase upstream/master

cat <<'EOF'

== done ==
next:
  scripts/sync-upstream.sh already stopped on any conflict; resolve, then:
    git add <files> && git rebase --continue
  build + test before moving on:
    cmake --build build-<dir> --target llama-server
  review the fork's whole delta after the rebase:
    git diff upstream/master
EOF
