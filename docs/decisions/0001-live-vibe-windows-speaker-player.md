---
status: accepted
date: 2026-10-03
---

# 0001 - Dedicated Windows speaker player for live-vibe under WSL

## Context and problem statement

Under WSL2, WSLg carries audio to Windows over RDP. live-vibe's synthesized speech left WSL clean (PulseAudio's
`RDPSink.monitor` matched a fresh Kokoro render) yet played on Windows full of crackle. The RDP leg adds the
crackle, so no fix on the Linux side removes it. How should the sidecar, running in WSL, get speech to the Windows
speaker without that leg?

## Decision drivers

- Clean speech on the Windows output device.
- The echo canceller needs a time-aligned reference: which samples reached the device, and when, on a clock the
  sidecar can map onto the mic stream's clock.
- Barge-in must cut playback within a few milliseconds.
- No network exposure: no port, listener or firewall rule.
- Nothing installed globally on Windows; the player must not outlive the sidecar.
- The mic path (WSLg) and the Linux TTS stay as they are.

## Considered options

1. Stay on WSLg/PulseAudio audio.
2. A Windows-side localhost service that the sidecar sends PCM to.
3. Windows-side TTS (synthesize on Windows instead of in WSL).
4. A player process on Windows, launched through WSL interop, fed PCM over its stdin pipe.

## Decision outcome

Chosen option: 4, a dedicated player on Windows fed over a stdin pipe. The sidecar (`bin/sidecar/winplayer.py`)
launches `bin/sidecar/win_player.py` through WSL interop and streams PCM and JSON control frames over its stdin; the
player reports `hello`, `open`, `played`, `end`, `cut`, `pong` and `bye` as JSON lines on stdout and plays through
WASAPI. With `speakerBackend` on `auto` (the default) this is the speaker under WSL with interop; `local` keeps the
WSLg path. The mic stays on WSLg. Everything on Windows lives in `%LOCALAPPDATA%\live-vibe`. Shipped in commit
8cc0216 (`feat(live-vibe): play speech on Windows under WSL`, 2026-10-03), released as live-vibe 0.5.0.

Option 1 is rejected on the crackle measurement. Option 2 is rejected because a pipe needs no network, port or
firewall rule (the property the design states). Option 3 is rejected (inferred, not stated in the sources): it
would move the synthesizer and its models to a second platform while the echo reference still has to come from
the device on Windows.

## Consequences

Positive:

- Playback goes straight to WASAPI, bypassing the RDP leg.
- Echo alignment holds: the player reports, per device callback, the samples it handed to WASAPI and their DAC time
  on its perf counter; ping/pong maps that clock onto the sidecar's (lowest round trip of the last 32, measured
  0.24 ms). What stays unreported (latency past WASAPI such as Bluetooth, the room, the RDP leg of the mic) only
  makes the reference lead the echo, and AEC3's delay estimator absorbs that lead.
- Barge-in cuts within a pipe round trip (measured: play() returns 5-6 ms after the cancel).
- No network surface; the player exits on stdin EOF, so it ends with the sidecar.

Negative:

- A second runtime on Windows: a pinned, sha256-checked `uv.exe` (0.10.8), uv's cache and a Windows Python with
  numpy and sounddevice, about 60 MB on first `/live setup`.
- Cold-start cost: every session start launches that runtime before the first sample plays.
- A handshake with timeouts and a watchdog: `hello` within 45 s, then `open` within 10 s, then clock sync; the
  player exits after 15 s without a frame. Any failure falls back to the WSLg speaker with one warning, and a
  mid-session death hands playback to WSLg.

## Amendment (2026-10-03)

The player implementation is moving from a uv-run Python script to a Rust single binary. In two sessions on
2026-10-03 (live-vibe 0.5.3, sidecar log 13:19 and 13:22) the device open did not complete within the sidecar's
10 s open wait; the sidecar logged "the device did not open", the player then exited on its watchdog ("the sidecar
went quiet", 0 frames), and each session stayed on WSLg with no retry. The delay is attributed to the Python cold
start: uv.exe, Windows Python, the numpy import and PortAudio's device scan (the log records the failure, not a
per-step breakdown). An earlier session that day opened the device about 4 s after launch. The Rust port keeps the
same stdin protocol, so the sidecar side and the decision above are unchanged.
