# Recording the live-vibe demo

The root README has a slot for a short clip of `/livevibe` in use. This page is the recipe for recording it. The
clip needs a real mic, speaker or headset, and the GPU, so it is recorded by hand.

Target length: 30 to 45 s. Use headphones so the clip has no echo, and say one thing at a time.

## Tools

- **Audio and terminal together (preferred):** a Windows screen recorder that captures system audio and the mic,
  such as OBS Studio or the Xbox Game Bar (Win+G). Record only the terminal window. Export as mp4 (H.264, AAC).
- **Terminal only:** `vhs` or `asciinema` (with `agg` to make a GIF). These capture no audio, so use them only for
  a silent fallback and add captions for what was said.

## Steps

1. **Start clean.** Run `/live setup` once beforehand so nothing downloads on camera. Open a fresh Claude Code
   session in a small repo, clear the screen, and start recording.
2. **Start live vibe.** Type `/livevibe`. Wait for the status line to leave `loading`. Show the start line.
3. **Ask one spoken question.** Say: "What does the main function in this repo do?" The transcript shows a
   `live-vibe: you (voice): ...` line and the front's `spoken: Let me check.` acknowledgement.
4. **Show the collapsed front lines.** Hold on the `front prompt (3 lines, ctrl+o to expand): task: ...` line for a
   second, then press ctrl+o once to show the full prompt the front handed Claude, and press it again to collapse it.
5. **Show Claude's answer.** Let Claude's answer print and the front read its `spoken summary: ...` aloud.
6. **Barge in.** Ask a follow-up that gets a longer answer, such as "Walk me through the tests." While the voice is
   reading the summary, say "Stop, that's enough." The voice stops on your first word. Stop recording a second later.

Trim dead air at the start and end. Keep the terminal font large enough to read at 720p.

## Adding it to the README

Commit the clip as `docs/demo/live-vibe-demo.mp4` (keep it under 10 MB). GitHub does not play a relative mp4 link
inline, so upload the file once through the GitHub web editor (drag it into the README edit box on github.com), which
gives a `https://github.com/user-attachments/assets/<id>` URL, and paste that URL on its own line under `## Demo`:

```markdown
## Demo

https://github.com/user-attachments/assets/<id>

The same clip is in the repo at [docs/demo/live-vibe-demo.mp4](docs/demo/live-vibe-demo.mp4).
```

For a GIF instead (silent, from `vhs` or `agg`), commit `docs/demo/live-vibe-demo.gif` and use:

```markdown
## Demo

![live-vibe: a spoken question, Claude's answer and a barge-in](docs/demo/live-vibe-demo.gif)
```
