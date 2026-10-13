# ottomation

ottomation is a Claude Code plugin marketplace. Its first plugin is live-vibe, which adds voice and director modes
to Claude Code:

- `/live` is full-duplex voice with Claude. The mic stays open, and you can talk over an answer to cut it off.
- `/vibe` is director mode. Claude reads and directs worker subagents instead of editing files itself.
- `/livevibe` lets you talk with a small, fast voice model that hands real work to Claude in vibe mode and tells you
  the result.

## Install

```
/plugin marketplace add potto007/ottomation
/plugin install live-vibe@ottomation
```

Then run `/live setup` once on each machine.

## Requirements

Claude Code 2.1.287 or newer, uv, a mic and a speaker; see the
[plugin README](plugins/live-vibe/README.md#requirements) for platform details.

## Documentation

- [Plugin README](plugins/live-vibe/README.md)
- [ADR 0001: Dedicated Windows speaker player for live-vibe under WSL](docs/decisions/0001-live-vibe-windows-speaker-player.md)
- [ADR 0002: ottomation is live-vibe's single home; branches are tried from a folder marketplace](docs/decisions/0002-ottomation-single-home-and-folder-marketplace.md)

## Developing

This repo is live-vibe's only home. Work happens on a branch in a worktree, and a branch is tried live before it is
merged, without a release and without touching the installed plugin:

```sh
git worktree add .claude/worktrees/<topic> -b <branch>
claude plugin marketplace add /abs/path/to/.claude/worktrees/<topic>   # the ottomation marketplace now points here
claude plugin update live-vibe@ottomation
```

A marketplace is named by its `marketplace.json`, so adding the worktree re-points `ottomation` from GitHub to that
folder (the command says so), and the installed `live-vibe@ottomation` updates from it. A plugin installed from a
folder marketplace is read straight from the folder: edit, then `/reload-plugins`. `claude plugin list` shows
`Read from:` the worktree while this is on. Point it back when done:

```sh
claude plugin marketplace add potto007/ottomation
claude plugin update live-vibe@ottomation
```

Checks, from `plugins/live-vibe/`:

| Check | Needs | Where |
|---|---|---|
| `uvx ruff check bin/sidecar` | nothing | CI and local |
| `uv run --script bin/sidecar/main.py --unit` | nothing (a second) | CI and local |
| `claude plugin validate --strict .` and `claude plugin test .` | Claude Code | CI and local |
| `uv run --script bin/sidecar/main.py --selftest` | the GPU (Kokoro, Kyutai) | local, behind the GPU gate |
| `bin/sidecar/eval/relay_eval.sh`, `eval/microturn_bench.py` | a llama-server on the GPU | local; results go in the PR |
| a mic session from the dev install | mic, speaker, GPU | local; the sidecar log goes in the PR |

CI (`.github/workflows/ci.yml`) runs the first three rows on every pull request and checks that `plugin.json` and
`marketplace.json` carry the same version. A release is a merge to `main` with both versions bumped; users get it with
`claude plugin update live-vibe@ottomation`.

## License

MIT. See [LICENSE](LICENSE).
