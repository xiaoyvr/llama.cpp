# Maintaining this fork against upstream

This repository is a fork of `ggml-org/llama.cpp`. The fork is maintained as a
**patch series applied on top of upstream**: `upstream/master` is always the
base, and everything this fork adds sits on top of it. The integration branch is
the branch you build and run.

## Model

```
upstream/master          <- fetch often, never commit here
        |
        +-- fork patch series (rebased on top)
                |  strix: HIP kernels / qwen4exp
                |  fork:  shared MTP head, startlux decision type, ...
                +-- <integration branch>   (what you build and run)
```

Rebase (not merge) is deliberate. With upstream as the base, every upstream
change is present by construction, and every **conflict is exactly a place where
upstream touched code the fork also touched** - which is the review point: "did
upstream supersede this fork patch?". A merge would bury those decisions in one
big resolution and the fork would slowly drift from upstream.

## Branch layout

| branch | role |
|---|---|
| `master` | clean mirror of `upstream/master` (fast-forward only) |
| `<integration>` | upstream + the rebased fork patch series; what you use |
| feature branches | small deltas off `<integration>`, rebased after each sync |

Keep `master` current so the merge base is the newest upstream:

```bash
git fetch upstream
git switch master && git merge --ff-only upstream/master
```

## Conflict rule

For every conflict, decide one of three things and move on:

- **upstream now does this** -> drop the fork patch (this is how the series shrinks).
- **both sides need parts** -> keep both.
- **the fork still owns this** -> keep the fork side; note it as a known conflict.

`rerere` replays prior resolutions, so a known conflict resolves itself next time.

## Sync procedure

One command (fetch, audit, rebase; stops on conflicts):

```bash
scripts/sync-upstream.sh            # current branch
scripts/sync-upstream.sh strix-all  # or by name
```

What it does, and the manual equivalent:

```bash
git fetch upstream

# 1. see what upstream landed that overlaps the fork (do not skip this)
git log --oneline $(git merge-base HEAD upstream/master)..upstream/master

# 2. rebase the fork onto the new upstream
git rebase upstream/master

# 3. on conflicts: resolve, then
git add <files> && git rebase --continue

# 4. build + test (see below)
```

After the rebase, review the entire fork delta:

```bash
git diff upstream/master            # everything this fork adds
```

Any hunk that now exists only because upstream already does it is a candidate to
delete (drop the corresponding fork commit and rebase again).

## Why rebase in a worktree

So the checkout you serve from is never mid-rebase:

```bash
git worktree add ../sync upstream/master
# work in ../sync, build and test, then move the real branch when green:
git reset --hard <sync-branch>
git worktree remove ../sync
```

## Required git config (one time)

```bash
git config rerere.enabled true
git config rerere.autoupdate true
```

## Build + test after a sync

```bash
# build (example flags for Strix Halo / gfx1151)
cmake -S . -B build-strixall -G Ninja ... -DQWEN4EXP_QSA=ON
cmake --build build-strixall --parallel --target llama-server

# decision-model checks: see scripts/startlux-patch-gguf.py and the local harness
```

## Fork features and the files that overlap upstream

Keep fork edits in these files small and well commented - they are the recurring
conflict points:

| area | files |
|---|---|
| decision models | `common/common.{h,cpp}`, `tools/server/server-decision.{h,cpp}` |
| Qwen4Exp / memory | `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.{h,cpp}`, `src/models/models.h` |
| scheduling / kernels | `ggml/src/ggml-cuda/*`, `ggml/src/ggml-backend*.cpp` |
| MTP | `common/speculative.cpp`, `src/llama-context.cpp` |

Fork-only features and where they live:

- `startlux` decision type: 5 lines in `common/common.*` + `server-decision.*`,
  plus `scripts/startlux-patch-gguf.py` to add the `systemone` template,
  `decision.type` and temperatures to a StartLux GGUF.
- shared MTP head: `src/models/qwen4exp.cpp`, `src/llama-context.cpp`,
  `common/speculative.cpp`.

Prefer **metadata and separate files** over edits to shared upstream code: a
GGUF metadata patch never conflicts, and a new file never conflicts. Keep the
in-tree C++ delta as small as possible.
