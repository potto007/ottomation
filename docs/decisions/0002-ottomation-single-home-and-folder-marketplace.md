---
status: accepted
date: 2026-10-12
---

# 0002 - ottomation is live-vibe's single home; branches are tried from a folder marketplace

## Context and problem statement

live-vibe was developed in a private marketplace repo and mirrored here by hand. The two trees were identical at
0.7.0, but a feature merged in the private repo (the micro-turn front, `microturnMs`) could not be tried from the
installed plugin: Claude Code runs the copy it fetched at install time, and `claude plugin update` moves only when
the version on record changes. The mirror also meant two places to review, two histories, and a version that could
drift between them without anyone noticing. Where should live-vibe live, and how is a branch tried on a real mic
before it is merged and released?

## Decision drivers

- One source of truth: one history, one review queue, one README.
- A branch must be runnable live (mic, speaker, GPU) without a release and without touching the public install.
- The checks that need no GPU run on every pull request, on a machine nobody has to keep.
- GPU checks (self-test, relay eval, micro-turn bench, a mic session) stay local and behind the GPU gate.
- A release is one visible step, and the two version fields cannot disagree.

## Considered options

1. Keep the private repo as the dev copy and mirror releases here.
2. Develop here; try branches with `claude --plugin-dir <worktree>/plugins/live-vibe`.
3. Develop here; try branches by adding a worktree as a *folder* marketplace and installing from it.

## Decision outcome

Option 3. ottomation is the only home of live-vibe; the private copy is removed once this repo carries the port.

A plugin installed from a marketplace that was added as a folder, with a relative `source`, is read from that folder
itself, not from an installed copy: an edit followed by `/reload-plugins` runs in the session, with no version bump
and no reinstall. So a branch is tried like this:

```sh
git worktree add .claude/worktrees/<topic> -b <branch>
claude plugin marketplace add /path/to/.claude/worktrees/<topic>     # once per worktree; name it ottomation-dev
claude plugin install live-vibe@ottomation-dev
```

with `live-vibe@ottomation` set to `false` in `enabledPlugins` for the duration, so the two do not both register
`/live`. The public install is untouched and comes back by flipping the flag.

`--plugin-dir` (option 2) is kept for a one-off session, but it does not disable the installed plugin and so loads
both; the folder marketplace is the documented route.

CI (`.github/workflows/ci.yml`) runs on every pull request, on Ubuntu, without a GPU: ruff on the sidecar, the
sidecar's `--unit` checks with the light dependency set they actually import, `claude plugin validate --strict`, the
mod's `*.test.ts` through `claude plugin test`, and a check that `plugin.json` and `marketplace.json` carry the same
version. GPU checks are run locally and recorded in the pull request as evidence.

A release is a merge to `main` with both versions bumped in the same pull request; users receive it through
`claude plugin update`.

## Consequences

- Good: one history (this repo already holds all of live-vibe's commits), one review queue, and a live trial of
  any branch in two commands.
- Good: a 0.7.0-twice situation (two different trees under one version) fails CI instead of surprising a user.
- Bad: `.claude/worktrees/` becomes a working convention of the repo (ignored in `.gitignore`); a stale worktree left
  registered as a marketplace keeps serving old code until it is removed with `claude plugin marketplace remove`.
- Bad: the CI dependency set is a second list beside the sidecar's PEP 723 header. An import the unit checks newly
  need fails CI until it is added to the workflow; that is the intended signal.
