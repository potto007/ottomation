# live-vibe

A Claude Code plugin marketplace with one plugin, live-vibe. It adds voice and director modes to Claude Code:

- `/live` is full-duplex voice with Claude. The mic stays open, and you can talk over an answer to cut it off.
- `/vibe` is director mode. Claude reads and directs worker subagents instead of editing files itself.
- `/livevibe` lets you talk with a small, fast voice model that hands real work to Claude in vibe mode and tells you
  the result.

## Install

```
/plugin marketplace add potto007/live-vibe
/plugin install live-vibe@potto007
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
