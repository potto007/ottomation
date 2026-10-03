"""--unit checks for the Windows player (winplayer.py, win_player.exe and win_player.py): the backend choice, the
command lines (the exe's, and uv's with its WSLENV), staging the exe, the pinned uv.exe unpacking, the Python
player's engine (clips, ends, cancel, continuous DAC times; the Rust engine has the same as cargo tests), and the
protocol end to end with --fake, a timer in place of a device: against win_player.py run by this Python, and against
the Rust player built for this host (win_player/build.sh or cargo build) when there is one. End to end: open, clock
sync, a full clip feeding the echo reference and guard, a cancel, the heartbeat from hello on, the process life
(stdin EOF, quit, the watchdog, a SIGKILLed parent) and the fallbacks (no start, a player dying mid-session, the one
background retry). No Windows, device or network."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import protocol, winplayer
from .echo import APM_RATE, EchoGuard, EchoReference


PYTHON_PLAYER = [sys.executable, str(winplayer.SCRIPT)]
CRATE = Path(__file__).resolve().parents[2] / "win_player"


def rust_player() -> list[str] | None:
    """The Rust player built for this host: $LIVE_VIBE_WIN_PLAYER, else the crate's release or debug build."""
    env = os.environ.get("LIVE_VIBE_WIN_PLAYER")
    candidates = [Path(env)] if env else [CRATE / "target" / "release" / "win_player",
                                          CRATE / "target" / "debug" / "win_player"]
    found = next((c for c in candidates if c.is_file() and os.access(c, os.X_OK)), None)
    return [str(found)] if found else None


def fake_proc(*extra: str, player: list[str] = PYTHON_PLAYER) -> subprocess.Popen[bytes]:
    return subprocess.Popen([*player, "--fake", *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("win_player_under_test", winplayer.SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class RecordingReference:
    def __init__(self) -> None:
        self.blocks: list[tuple[float, int]] = []

    def played_at(self, block: np.ndarray, t: float) -> None:
        self.blocks.append((t, len(block)))


class Sent:
    def __init__(self) -> None:
        self.msgs: list[dict[str, Any]] = []

    def send(self, **obj: Any) -> None:
        self.msgs.append(obj)


def gone_within(proc: subprocess.Popen[bytes], s: float) -> bool:
    try:
        proc.wait(timeout=s)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False


def winplayer_units(check: Callable[[Any, str], bool]) -> None:
    emitted: list[dict[str, Any]] = []  # the log and warn lines, kept off the test output
    protocol.capture(emitted)
    try:
        _units(check, emitted)
    finally:
        protocol.capture(None)


def _units(check: Callable[[Any, str], bool], warned: list[dict[str, Any]]) -> None:
    # -- backend choice
    def resolve(backend: str, wsl: bool, problem: str | None) -> tuple[str, str]:
        real = winplayer.under_wsl, winplayer.interop_problem
        winplayer.under_wsl, winplayer.interop_problem = (lambda: wsl), (lambda: problem)
        try:
            return winplayer.resolve(backend)
        finally:
            winplayer.under_wsl, winplayer.interop_problem = real

    got = [resolve("auto", True, None)[0], resolve("auto", False, None)[0], resolve("auto", True, "off")[0],
           resolve("local", True, None)[0], resolve("windows", True, None)[0], resolve("windows", False, None)]
    check(got[:5] == ["windows", "local", "local", "local", "windows"] and got[5][0] == "local"
          and "needs" not in got[5][1] and "not WSL" in got[5][1],
          f"speaker backend: auto is Windows only under WSL with interop; windows off WSL falls back {got}")

    # -- command line
    paths = winplayer.WinPaths("C:\\Users\\u\\AppData\\Local\\live-vibe", Path("/mnt/c/Users/u/AppData/Local/live-vibe"))
    real_env = os.environ.get("WSLENV")
    os.environ["WSLENV"] = "USERPROFILE/p"
    try:
        argv, env = winplayer.command(paths, Path("/mnt/c/x/uv.exe"), paths.win("win_player.py"), "Stealth", 0.05)
    finally:
        if real_env is None:
            os.environ.pop("WSLENV", None)
        else:
            os.environ["WSLENV"] = real_env
    check(argv[:4] == ["/mnt/c/x/uv.exe", "run", "--script", "C:\\Users\\u\\AppData\\Local\\live-vibe\\win_player.py"]
          and argv[-2:] == ["--speaker", "Stealth"] and env["UV_CACHE_DIR"].endswith("live-vibe\\uv-cache")
          and env["WSLENV"].startswith("USERPROFILE/p:") and "UV_PYTHON_INSTALL_DIR" in env["WSLENV"].split(":"),
          f"command: uv.exe run --script, caches under live-vibe, passed through WSLENV ({env['WSLENV']})")
    argv = winplayer.exe_command(Path("/mnt/c/x/live-vibe/win_player-ab.exe"), "Stealth", 0.05)
    check(argv == ["/mnt/c/x/live-vibe/win_player-ab.exe", "--latency", "0.05", "--speaker", "Stealth"]
          and winplayer.exe_command(Path("/x.exe"), "", 0.05) == ["/x.exe", "--latency", "0.05"],
          f"command: the exe runs directly through interop, no uv {argv}")

    # -- staging the exe: named by its content, copied once, older copies removed
    with tempfile.TemporaryDirectory() as d:
        sp = winplayer.WinPaths("C:\\x", Path(d) / "live-vibe")
        src = Path(d) / "win_player.exe"
        real_exe = winplayer.EXE
        winplayer.EXE = src
        try:
            src.write_bytes(b"MZ one")
            first = winplayer.stage_exe(sp)
            mtime = first.stat().st_mtime_ns
            again = winplayer.stage_exe(sp)
            kept = again.stat().st_mtime_ns == mtime
            src.write_bytes(b"MZ two, rebuilt")
            second = winplayer.stage_exe(sp)
        finally:
            winplayer.EXE = real_exe
        left = sorted(x.name for x in sp.root.iterdir())
        check(first == again and kept and second != first
              and second.read_bytes() == b"MZ two, rebuilt" and left == [second.name]
              and second.name.startswith("win_player-"),
              f"stage: the exe is copied once under a content name and a rebuild replaces it {left}")
    check(winplayer.EXE.is_file() and winplayer.EXE.read_bytes()[:2] == b"MZ",
          f"stage: the plugin ships a prebuilt {winplayer.EXE.name} (win_player/build.sh)")

    # -- the pinned uv.exe: downloaded sha256-checked (front_server.download), unpacked, archive removed
    from . import front_server

    with tempfile.TemporaryDirectory() as d:
        p = winplayer.WinPaths("C:\\x", Path(d))
        asked: list[tuple[str, str, int]] = []

        def fake_download(url: str, dest: Path, sha: str, size: int, progress: Any, label: str, stop: Any = None) -> Path:
            asked.append((url, sha, size))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("uv.exe", b"MZ fake uv")
                z.writestr("uvx.exe", b"MZ fake uvx")
            return dest

        real_dl, real_which = front_server.download, winplayer.shutil.which
        front_server.download, winplayer.shutil.which = fake_download, (lambda _name: None)
        try:
            uv = winplayer.ensure_uv(p, lambda _t: None)
            again = winplayer.ensure_uv(p, lambda _t: None)
        finally:
            front_server.download, winplayer.shutil.which = real_dl, real_which
        name, sha, size = winplayer.uv_asset()
        check(uv == again == p.pinned_uv and uv.read_bytes() == b"MZ fake uv" and len(asked) == 1
              and asked[0] == (winplayer.UV_RELEASE + name, sha, size)
              and not list((Path(d) / "uv" / "downloads").glob("*.zip")),
              f"uv.exe: the pinned release with its sha256, unpacked once into the cache {asked}")

    # -- the engine, in process
    wp = load_script()
    out = Sent()
    eng = wp.Engine(out, 1000)
    eng.add(1, 30, 0, np.full(20, 0.1, np.float32))
    eng.add(1, 30, 20, np.full(10, 0.2, np.float32))
    eng.add(2, 15, 0, np.full(15, 0.3, np.float32))
    buf = np.ones(40, np.float32)
    eng.fill(buf, 5.0, False)
    played = [(m["id"], m["at"], m["n"], m["dac"]) for m in out.msgs if m["t"] == "played"]
    ends = [(m["id"], m["n"]) for m in out.msgs if m["t"] == "end"]
    check(played == [(1, 0, 20, 5.0), (1, 20, 10, 5.02), (2, 0, 10, 5.03)] and ends == [(1, 30)]
          and (buf[:20] == np.float32(0.1)).all() and (buf[30:] == np.float32(0.3)).all(),
          f"engine: clips back to back in one block, each run reported with its DAC time {played} {ends}")
    out.msgs.clear()
    eng.fill(buf, 5.0401, True)  # 0.1 ms off the running clock: followed, not re-anchored
    first = [m for m in out.msgs if m["t"] == "played"][0]
    check(abs(first["dac"] - 5.04) < 1e-4 and not buf[5:].any() and ("end" in [m["t"] for m in out.msgs])
          and eng.xruns == 1, f"engine: continuous DAC times, silence after the last clip, underflows counted {first}")
    eng.add(3, 100, 0, np.full(50, 0.4, np.float32))
    eng.fill(np.zeros(10, np.float32), 6.0, False)
    cut = eng.cancel(3)
    eng.add(3, 100, 50, np.full(50, 0.4, np.float32))  # a late chunk of the cancelled clip
    tail = np.ones(10, np.float32)
    eng.fill(tail, 6.01, False)
    check(cut == 10 and not eng.clips and not tail.any(),
          f"engine: cancel reports what played and drops late chunks ({cut})")

    # -- end to end against the fake player: the Python one, and the Rust one when built for this host
    end_to_end(check, "fake player", PYTHON_PLAYER)
    rust = rust_player()
    if rust is not None:
        end_to_end(check, "fake rust player", rust)
    else:
        check(True, "fake rust player: not built for this host (win_player/build.sh); its end-to-end checks skipped")

    # -- fallbacks
    warned.clear()
    real_paths = winplayer.win_paths
    try:
        def no_paths() -> winplayer.WinPaths:
            raise winplayer.WinPlayerError("cannot read %LOCALAPPDATA%")

        winplayer.win_paths = no_paths
        got_player = winplayer.Launch().start().player(24_000, timeout=5)
    finally:
        winplayer.win_paths = real_paths
    warns = [w["text"] for w in warned if w["type"] == "warn"]
    check(got_player is None and len(warns) == 1 and "LOCALAPPDATA" in warns[0] and "WSLg" in warns[0],
          f"fallback: a player that cannot start is one warn, and the local speaker plays {warns}")

    clip = np.full(8000, 0.25, np.float32)
    proc = fake_proc()
    p4 = winplayer.WinPlayer(proc)
    p4.open(16_000, timeout=20)

    class Local:
        played: list[int] = []

        def play(self, audio: np.ndarray, cancel: threading.Event) -> int:
            self.played.append(len(audio))
            return len(audio)

        def close(self) -> None:
            pass

    local = Local()
    p4.fallback = lambda: local
    proc.kill()
    proc.wait()
    warned.clear()
    try:
        first_n = p4.play(clip, threading.Event())
        second_n = p4.play(clip[:100], threading.Event())
    finally:
        p4.close()
    warns = [w["text"] for w in warned if w["type"] == "warn"]
    check(first_n == 8000 and second_n == 100 and local.played == [100] and len(warns) == 1 and "WSLg" in warns[0],
          f"fallback: a player that dies mid-session warns once and the local speaker takes over {warns}")

    speaker_retry(check)

    # -- the echo reference on a remote player's clock
    er = EchoReference(16_000)
    mono = time.monotonic()
    er.take(160, 1000.0 + mono + 0.05, 16_000, now=1000.0 + mono)  # the mic stream's clock runs 1000 s ahead
    er.played_at(np.full(1600, 0.5, np.float32), mono + 0.05)
    got_ref = er.take(800, 1000.0 + mono + 0.06, 16_000)
    check(abs(er.pa_offset - 1000.0) < 0.01 and np.abs(got_ref - 0.5).max() < 1e-3 and APM_RATE == 16_000,
          f"echo reference: played_at lands on the mic stream's clock (offset {er.pa_offset:.3f})")


def end_to_end(check: Callable[[Any, str], bool], label: str, player: list[str]) -> None:
    clip = np.full(8000, 0.25, np.float32)
    proc = fake_proc(player=player)
    p2 = winplayer.WinPlayer(proc)
    ref, guard = RecordingReference(), EchoGuard(lambda: False)
    guard.sound_until = 0.0
    t = time.monotonic()
    opened = p2.open(16_000, ref, guard, timeout=20)
    took = time.monotonic() - t
    check(opened and p2.info.get("device") == "fake" and p2.offset is not None and p2.rtt < 0.05,
          f"{label}: opens and syncs its clock ({took:.2f}s, rtt {1000 * p2.rtt:.2f} ms)")
    t = time.monotonic()
    n = p2.play(clip, threading.Event())
    took = time.monotonic() - t
    total = sum(k for _, k in ref.blocks)
    steps = [ref.blocks[i + 1][0] - ref.blocks[i][0] - ref.blocks[i][1] / 16_000 for i in range(len(ref.blocks) - 1)]
    check(n == 8000 and 0.4 < took < 0.8 and total == 8000 and max(map(abs, steps), default=1) < 2e-3 and guard.active(),
          f"{label}: a 0.5 s clip plays in real time ({took:.2f}s) and feeds the reference contiguously and the guard")
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t = time.monotonic()
    n = p2.play(np.full(32_000, 0.25, np.float32), cancel)
    took = time.monotonic() - t
    check(4000 <= n <= 6400 and took < 0.3 + 0.08,
          f"{label}: a cancel at 0.3 s cuts within 80 ms ({n} samples, {took:.3f}s)")
    n = p2.play(clip[:1600], threading.Event())
    check(n == 1600, f"{label}: plays on after a cancel")
    p2.close()
    check(p2.bye.get("why") == "quit" and proc.poll() == 0, f"{label}: quit ends it ({p2.bye})")

    proc = fake_proc(player=player)
    p3 = winplayer.WinPlayer(proc)
    p3.open(16_000, timeout=20)
    proc.stdin.close()  # type: ignore[union-attr]
    check(gone_within(proc, 3.0), f"{label}: stdin EOF ends it")

    proc = fake_proc("--watchdog", "0.6", player=player)  # no pings: nothing reads it, so no heartbeat runs
    check(gone_within(proc, 4.0), f"{label}: the watchdog ends it when the sidecar goes quiet")
    proc.stdin.close()  # type: ignore[union-attr]

    proc = fake_proc("--watchdog", "1.5", player=player)
    p5 = winplayer.WinPlayer(proc)  # read from the start, as Launch does, but not opened yet (the models load)
    time.sleep(3.5)
    alive = proc.poll() is None
    opened = p5.open(16_000, timeout=5)
    p5.close()
    check(alive and opened, f"{label}: the heartbeat runs from hello, so a player not yet opened outlives its watchdog")

    holder = ("import json, subprocess, sys, time; p = subprocess.Popen(json.loads(sys.argv[1]) + ['--fake'], "
              "stdin=subprocess.PIPE, stdout=subprocess.PIPE); print(p.pid, flush=True); time.sleep(60)")
    h = subprocess.Popen([sys.executable, "-c", holder, json.dumps(player)], stdout=subprocess.PIPE, text=True)
    child = int(h.stdout.readline())  # type: ignore[union-attr]
    os.kill(h.pid, signal.SIGKILL)
    h.wait()
    deadline = time.monotonic() + 3.0
    alive = True
    while alive and time.monotonic() < deadline:
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/stat") as f:
                alive = f.read().split()[2] != "Z"
        except (OSError, FileNotFoundError):
            alive = False
        time.sleep(0.05)
    check(not alive, f"{label}: a SIGKILLed parent leaves no player behind (stdin EOF)")


def speaker_retry(check: Callable[[Any, str], bool]) -> None:
    """Speaker: after a failure, one background retry that takes over at the next play()."""

    class Local:
        def __init__(self) -> None:
            self.played: list[int] = []
            self.closed = False

        def play(self, audio: np.ndarray, cancel: threading.Event) -> int:
            self.played.append(len(audio))
            return len(audio)

        def close(self) -> None:
            self.closed = True

    # at start: the local speaker plays, the retry brings the Windows player up, the next sentence plays there
    local = Local()
    calls: list[int] = []
    procs: list[subprocess.Popen[bytes]] = []

    def retry() -> winplayer.WinPlayer | None:
        calls.append(1)
        proc = fake_proc()
        procs.append(proc)
        p = winplayer.WinPlayer(proc)
        return p if p.open(16_000, timeout=20) else None

    sp = winplayer.Speaker(local, retry, delay=0.0)
    first = sp.play(np.zeros(160, np.float32), threading.Event())
    by = time.monotonic() + 10
    while sp._next is None and time.monotonic() < by:
        time.sleep(0.02)
    second = sp.play(np.zeros(1600, np.float32), threading.Event())
    switched = isinstance(sp.current, winplayer.WinPlayer)
    sp.close()
    check(first == 160 and second == 1600 and local.played == [160] and local.closed and switched and calls == [1]
          and procs[0].wait(3) == 0,
          "retry: after a failed start the local speaker plays, one retry brings Windows back between sentences")

    # a retry that fails too: the local speaker keeps playing, and there is no second retry
    local = Local()
    calls.clear()
    sp = winplayer.Speaker(local, lambda: calls.append(1) or None, delay=0.0)  # type: ignore[func-returns-value]
    time.sleep(0.2)
    sp.play(np.zeros(10, np.float32), threading.Event())
    time.sleep(0.2)
    sp.play(np.zeros(10, np.float32), threading.Event())
    sp.close()
    check(calls == [1] and local.played == [10, 10], f"retry: one attempt a session ({len(calls)})")
