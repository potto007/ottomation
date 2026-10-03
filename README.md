# ottomation

ottomation is a Claude Code plugin marketplace. Its first plugin is live-vibe, which adds voice and director modes
to Claude Code:

- `/live` is full-duplex voice with Claude. The mic stays open, and you can talk over an answer to cut it off.
- `/vibe` is director mode. Claude reads and directs worker subagents instead of editing files itself.
- `/livevibe` lets you talk with a small, fast voice model that hands real work to Claude in vibe mode and tells you
  the result.

## Demo

A short clip of `/livevibe` is coming. Until then,
[What live vibe prints](plugins/live-vibe/README.md#what-live-vibe-prints) shows the lines a session prints around
Claude's answers. The recording recipe is in [docs/demo](docs/demo/README.md).

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

## Licence

MIT. See [LICENSE](LICENSE).
