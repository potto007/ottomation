"""Speech on Windows for a sidecar under WSL. WSLg carries audio over RDP, and on WSL2 that transport adds crackle
the Linux side never produced (measured: the RDPSink monitor matched a fresh Kokoro render). So the sidecar can play
through win_player.py on Windows instead: uv.exe runs it through WSL interop, and the PCM streams over its stdin, so
there is no network, port or firewall. The mic stays on WSLg.

Everything on Windows lives in %LOCALAPPDATA%\\live-vibe: a pinned uv.exe when none is on the PATH, uv's cache, any
Python uv downloads, and a copy of win_player.py. Nothing is installed globally.

Echo alignment: the player reports, per device callback, which samples of which clip it handed to WASAPI and the
DAC time of the first one on its perf counter (QueryPerformanceCounter; PortAudio's outputBufferDacTime, or the
callback time plus the stream latency where the host API gives none). Ping/pong maps that clock onto
time.monotonic() (offset of the lowest-round-trip exchange of the last 32, error within half that round trip, about
a millisecond over the interop pipe), and EchoReference.played_at moves it onto the mic stream's clock. The reference
then leads the echo by what no one reports: device and driver latency past WASAPI (a Bluetooth link: 100-250 ms),
the room, and the RDP leg of the mic (the WSL ADC time does not include it). That lead is never negative beyond the
sync error, and AEC3's delay estimator absorbs it (offline it held 150-600 ms)."""
from __future__ import annotations

import collections
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import IO, Any, Callable

import numpy as np

from .protocol import file_log, log, warn

SCRIPT = Path(__file__).with_name("win_player.py")
HEADER = struct.Struct("<cI")
AUDIO = struct.Struct("<III")
READY_TIMEOUT_S = 45.0  # a run whose uv cache is cold fetches Python and numpy first; /live setup does that ahead
SETUP_TIMEOUT_S = 600.0

# uv 0.10.8 for Windows; sha256 digests are GitHub's own, from the release API (assets[].digest).
UV_VERSION = "0.10.8"
UV_RELEASE = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
UV_ASSETS = {
    "x86_64": ("uv-x86_64-pc-windows-msvc.zip", "2e70ecd22196cbd9d14eefb700814bcafc5b75a0d8275b52e8402e5fe256d928",
               22159808),
    "aarch64": ("uv-aarch64-pc-windows-msvc.zip", "20db25dc446f9a75d1cfde0a5f4b021e1b2eb266e600a610d32c7ca5d7ff83bf",
                20632208),
}


class WinPlayerError(RuntimeError):
    pass


# -- where we run ---------------------------------------------------------------------------------------------
def under_wsl() -> bool:
    return sys.platform == "linux" and ("microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME")))


def interop_problem() -> str | None:
    """Why Windows programs cannot run from here, or None when they can."""
    for name in ("WSLInterop", "WSLInterop-late"):
        f = Path("/proc/sys/fs/binfmt_misc") / name
        if f.exists():
            try:
                if f.read_text().splitlines()[0].strip() != "enabled":
                    return "WSL interop is disabled"
            except (OSError, IndexError):
                pass
            break
    else:
        return "WSL interop is off (no WSLInterop binfmt entry)"
    if not _cmd_exe():
        return "cmd.exe not found (the Windows drives are not mounted)"
    return None


def resolve(backend: str) -> tuple[str, str]:
    """speakerBackend -> ("windows" | "local", why)."""
    if backend == "local":
        return "local", "speakerBackend local"
    wsl = under_wsl()
    why = interop_problem() if wsl else "not WSL"
    if wsl and why is None:
        return "windows", "WSL with interop" if backend == "auto" else "speakerBackend windows"
    if backend == "windows":
        return "local", f"speakerBackend windows, but {why}"
    return "local", why or "local"


def _cmd_exe() -> str | None:
    found = shutil.which("cmd.exe")
    if found:
        return found
    for p in (Path("/mnt/c/Windows/System32/cmd.exe"), Path("/mnt/c/windows/system32/cmd.exe")):
        if p.exists():
            return str(p)
    return None


class WinPaths:
    """%LOCALAPPDATA%\\live-vibe, as Windows sees it (win) and as this side does (root)."""

    def __init__(self, root_win: str, root: Path):
        self.root_win, self.root = root_win.rstrip("\\"), root

    def win(self, *parts: str) -> str:
        return "\\".join([self.root_win, *parts])

    @property
    def pinned_uv(self) -> Path:
        return self.root / "uv" / UV_VERSION / "uv.exe"


_paths: WinPaths | None = None


def win_paths() -> WinPaths:
    global _paths
    if _paths is not None:
        return _paths
    cmd = _cmd_exe()
    if not cmd:
        raise WinPlayerError("cmd.exe not found")
    # cwd on a Windows drive: from a Linux folder cmd.exe warns about UNC paths
    r = subprocess.run([cmd, "/d", "/c", "echo %LOCALAPPDATA%"], capture_output=True, text=True, timeout=20,
                       cwd=str(Path(cmd).parent))
    local = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
    if not local or "%" in local:
        raise WinPlayerError(f"cannot read %LOCALAPPDATA% ({r.stderr.strip()[:120] or 'empty'})")
    u = subprocess.run(["wslpath", "-u", local], capture_output=True, text=True, timeout=10)
    if u.returncode != 0 or not u.stdout.strip():
        raise WinPlayerError(f"wslpath cannot map {local!r}")
    _paths = WinPaths(local + "\\live-vibe", Path(u.stdout.strip()) / "live-vibe")
    return _paths


def find_uv(paths: WinPaths) -> Path | None:
    """uv.exe on the PATH (Windows' PATH reaches WSL), else the pinned copy in the cache."""
    found = shutil.which("uv.exe")
    if found:
        return Path(found)
    return paths.pinned_uv if paths.pinned_uv.is_file() else None


def uv_asset() -> tuple[str, str, int]:
    """The uv build for this machine: WSL's architecture is Windows' own."""
    return UV_ASSETS["aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"]


def ensure_uv(paths: WinPaths, progress: Callable[[str], None]) -> Path:
    found = find_uv(paths)
    if found:
        return found
    from .front_server import download

    name, sha, size = uv_asset()
    progress(f"Windows player: downloading uv {UV_VERSION} for Windows ({size / 1e6:.0f} MB)")
    z = download(UV_RELEASE + name, paths.root / "uv" / "downloads" / name, sha, size,
                 lambda t: progress(t.replace("front: ", "Windows player: ", 1)), name)
    with zipfile.ZipFile(z) as zf:
        member = next((m for m in zf.namelist() if m.rsplit("/", 1)[-1] == "uv.exe"), None)
        if member is None:
            raise WinPlayerError(f"{name} has no uv.exe")
        paths.pinned_uv.parent.mkdir(parents=True, exist_ok=True)
        tmp = paths.pinned_uv.with_suffix(".part")
        with zf.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        tmp.replace(paths.pinned_uv)
    os.chmod(paths.pinned_uv, 0o755)
    z.unlink(missing_ok=True)
    return paths.pinned_uv


def stage_script(paths: WinPaths) -> str:
    """A copy of win_player.py on the Windows drive (a \\\\wsl$ path works but is slow and fragile there)."""
    data = SCRIPT.read_bytes()
    dest = paths.root / "win_player.py"
    if not dest.exists() or dest.read_bytes() != data:
        paths.root.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return paths.win("win_player.py")


def command(paths: WinPaths, uv: Path, script: str, speaker: str, latency: float) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ)
    env.update(UV_CACHE_DIR=paths.win("uv-cache"), UV_PYTHON_INSTALL_DIR=paths.win("python"), UV_NO_PROGRESS="1")
    names = "UV_CACHE_DIR:UV_PYTHON_INSTALL_DIR:UV_NO_PROGRESS"  # WSLENV: what reaches the Windows process
    env["WSLENV"] = f"{env['WSLENV']}:{names}" if env.get("WSLENV") else names
    argv = [str(uv), "run", "--script", script, "--latency", str(latency)]
    if speaker:
        argv += ["--speaker", speaker]
    return argv, env


# -- the player -----------------------------------------------------------------------------------------------
class WinPlayer:
    """audio.Player's interface over win_player.py: play() blocks, stops within one pipe round trip of `cancel`
    (the player drops its queue and the device buffer), and returns how many samples were played. The reader thread
    turns the player's reports into the echo reference and the echo guard's tail, as the local callback does."""

    LATENCY_S = 0.05
    STALL_S = 2.0  # no report for this long while a clip should play: the player is gone
    PING_S = 1.0  # also the player's heartbeat: it exits after --watchdog (15 s) without a frame
    CHUNK = 12_000  # samples per audio frame
    CUT_WAIT_S = 0.3

    def __init__(self, proc: Any, label: str = "Windows player"):
        self.proc, self.label = proc, label
        self.sample_rate = 0
        self.reference: Any = None
        self.guard: Any = None
        self.fallback: Callable[[], Any] | None = None  # a local Player to switch to if this one dies
        self.info: dict[str, Any] = {}
        self.error = ""
        self.underflows = 0
        self.bye: dict[str, Any] = {}
        self.offset: float | None = None  # player perf counter minus time.monotonic()
        self.rtt = 0.0
        self._sync: collections.deque[tuple[float, float]] = collections.deque(maxlen=32)
        self._wlock = threading.Lock()
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._hello = threading.Event()
        self._opened = threading.Event()
        self._gone = threading.Event()
        self._id = 0
        self._clips: dict[int, np.ndarray] = {}
        self._progress: dict[int, int] = {}
        self._ended: set[int] = set()
        self._cuts: dict[int, tuple[int, float]] = {}
        self._last_report = 0.0
        self._broken = False
        self._delegate: Any = None
        self._closed = False
        threading.Thread(target=self._read, name="winplayer-out", daemon=True).start()
        threading.Thread(target=self._drain_stderr, name="winplayer-err", daemon=True).start()

    # -- wire
    def _send(self, kind: bytes, payload: bytes) -> bool:
        try:
            with self._wlock:
                self.proc.stdin.write(HEADER.pack(kind, len(payload)) + payload)
                self.proc.stdin.flush()
            return True
        except (OSError, ValueError, AttributeError):
            self._gone.set()
            return False

    def _send_json(self, **obj: Any) -> bool:
        return self._send(b"J", json.dumps(obj, separators=(",", ":")).encode())

    def _read(self) -> None:
        out: IO[bytes] = self.proc.stdout
        try:
            for raw in out:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    file_log("INFO", f"windows player: {raw.decode('utf-8', 'replace').rstrip()}")
                    continue
                self._on(msg)
        except (OSError, ValueError):
            pass
        finally:
            self._gone.set()
            with self._cv:
                self._cv.notify_all()

    def _drain_stderr(self) -> None:
        err = getattr(self.proc, "stderr", None)
        if err is None:
            return
        try:
            for raw in err:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    file_log("INFO", f"windows player: {line}")
        except (OSError, ValueError):
            pass

    def _on(self, msg: dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "played":
            self._on_played(msg)
        elif t == "end":
            with self._cv:
                self._ended.add(int(msg["id"]))
                self._cv.notify_all()
        elif t == "cut":
            with self._cv:
                self._cuts[int(msg["id"])] = (int(msg["played"]), float(msg.get("ms", 0.0)))
                self._cv.notify_all()
        elif t == "pong":
            now = time.monotonic()
            t0 = float(msg["t0"])
            self._sync.append((now - t0, float(msg["w"]) - (t0 + now) / 2))
            self.rtt, self.offset = min(self._sync)
        elif t == "xrun":
            self.underflows = int(msg.get("total", self.underflows + 1))
        elif t == "hello":
            self.info["pid"], self.info["python"] = msg.get("pid"), msg.get("python")
            self._hello.set()
        elif t == "open":
            self.info.update(msg)
            self._opened.set()
        elif t == "error":
            self.error = str(msg.get("text", ""))
            log(f"windows player: {self.error}")
            self._opened.set()
        elif t == "bye":
            self.bye = msg

    def to_mono(self, w: float) -> float:
        """A player perf-counter time on time.monotonic(); before any pong, a fixed estimate: now plus the buffer."""
        if self.offset is None:
            return time.monotonic() + float(self.info.get("latency", self.LATENCY_S))
        return w - self.offset

    def _on_played(self, msg: dict[str, Any]) -> None:
        cid, at, n = int(msg["id"]), int(msg["at"]), int(msg["n"])
        self._last_report = time.monotonic()
        clip = self._clips.get(cid)
        if clip is not None and n > 0:
            dac = self.to_mono(float(msg["dac"]))
            try:
                if self.reference is not None:
                    self.reference.played_at(clip[at:at + n], dac)
                if self.guard is not None:
                    self.guard.sounding(dac + n / max(1, self.sample_rate))
            except Exception as e:  # noqa: BLE001 - echo handling is an aid
                file_log("INFO", f"windows player: echo reference: {type(e).__name__}: {e}")
        with self._cv:
            self._progress[cid] = max(self._progress.get(cid, 0), at + n)

    # -- life
    def open(self, rate: int, reference: Any = None, guard: Any = None, timeout: float = READY_TIMEOUT_S) -> bool:
        """Waits for the process, opens the device at `rate`, syncs the clocks. False with self.error on failure."""
        self.sample_rate, self.reference, self.guard = rate, reference, guard
        deadline = time.monotonic() + timeout
        if not self._wait(self._hello, deadline):
            self.error = self.error or ("the process exited" if self._gone.is_set() else f"no answer within {timeout:.0f}s")
            return False
        self._send_json(op="open", rate=rate)
        if not self._wait(self._opened, min(deadline, time.monotonic() + 10.0)) or "device" not in self.info:
            self.error = self.error or "the device did not open"
            return False
        for _ in range(8):
            self._send_json(op="ping", t=time.monotonic())
            time.sleep(0.02)
        sync_by = time.monotonic() + 2.0
        while len(self._sync) < 4 and time.monotonic() < sync_by and not self._gone.is_set():
            time.sleep(0.01)
        if self.offset is None:
            log("windows player: no clock sync; the echo reference uses the stream latency instead")
        threading.Thread(target=self._heartbeat, name="winplayer-ping", daemon=True).start()
        return True

    def _wait(self, ev: threading.Event, deadline: float) -> bool:
        while not ev.wait(0.05):
            if self._gone.is_set() or time.monotonic() > deadline:
                return ev.is_set()
        return True

    def _heartbeat(self) -> None:
        while not self._gone.wait(self.PING_S):
            if not self._send_json(op="ping", t=time.monotonic()):
                return

    @property
    def broken(self) -> bool:
        return self._broken

    def summary(self) -> str:
        """"Windows player via WASAPI on <device>, <rate> Hz, <latency> ms"."""
        i = self.info
        api = "WASAPI" if "WASAPI" in str(i.get("hostapi")) else str(i.get("hostapi", "?"))
        return (f"Windows player via {api} on {i.get('device', '?')}, {i.get('rate', '?')} Hz, "
                f"{1000 * float(i.get('latency', 0)):.0f} ms")

    def describe(self) -> str:
        i = self.info
        return (f"{self.summary()} (device mix {i.get('device_rate', '?')} Hz, opened as {i.get('how', '?')}, "
                f"{i.get('note', '')}); Windows pid {i.get('pid')}, Python {i.get('python')}; clock sync "
                f"{1000 * self.rtt:.1f} ms round trip")

    # -- playback
    def play(self, audio: np.ndarray, cancel: threading.Event) -> int:
        with self._lock:
            if self._broken:
                return self._play_instead(audio, cancel)
            clip = np.ascontiguousarray(audio, dtype=np.float32)
            if not len(clip):
                return 0
            self._id += 1
            cid = self._id
            self._clips = {cid: clip}  # the reader thread looks up the playing clip here
            under = self.underflows
            sent_at = time.monotonic()
            ok = all(self._send(b"A", AUDIO.pack(cid, len(clip), i) + clip[i:i + self.CHUNK].tobytes())
                     for i in range(0, len(clip), self.CHUNK))
            try:
                with self._cv:
                    while ok:
                        if cid in self._ended:
                            return len(clip)
                        if cancel.is_set():
                            break
                        if self._gone.is_set():
                            raise WinPlayerError(self.error or "the process exited")
                        if time.monotonic() - max(sent_at, self._last_report) > self.STALL_S:
                            raise WinPlayerError(f"no playback report for {self.STALL_S:.0f}s")
                        self._cv.wait(0.01)
                if not ok:
                    raise WinPlayerError("the pipe to the player closed")
                return self._cut(cid)
            except WinPlayerError as e:
                self._break(str(e))
                played = self._progress.get(cid, 0)
                return played if cancel.is_set() else len(clip)
            finally:
                if self.underflows > under:
                    log(f"speaker: {self.underflows - under} underflow(s) in one sentence (Windows player)")

    def _cut(self, cid: int) -> int:
        """Cancel: the player drops the clip and its device buffer and says how much it had played."""
        self._send_json(op="cancel", id=cid)
        by = time.monotonic() + self.CUT_WAIT_S
        with self._cv:
            while cid not in self._cuts and cid not in self._ended and time.monotonic() < by and not self._gone.is_set():
                self._cv.wait(0.01)
            if cid in self._cuts:
                return self._cuts[cid][0]
            return self._progress.get(cid, 0)

    def _break(self, why: str) -> None:
        self._broken = True
        if self.fallback is not None:
            try:
                self._delegate = self.fallback()
                warn(f"Windows speaker failed ({why}); playing through WSLg from here on.")
                return
            except Exception as e:  # noqa: BLE001
                why = f"{why}; local fallback failed too: {type(e).__name__}: {e}"
        warn(f"Windows speaker failed ({why}); answers continue silently.")

    def _play_instead(self, audio: np.ndarray, cancel: threading.Event) -> int:
        if self._delegate is not None:
            return int(self._delegate.play(audio, cancel))
        step = int(max(1, self.sample_rate) * 0.03)
        for i in range(0, len(audio), step):
            if cancel.is_set():
                return i
            cancel.wait(min(step, len(audio) - i) / max(1, self.sample_rate))
        return len(audio)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        got = self._lock.acquire(timeout=1.0)
        try:
            self._send_json(op="quit")
            try:
                self.proc.stdin.close()
            except (OSError, ValueError, AttributeError):
                pass
            try:
                self.proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
                self.proc.kill()
            if self.bye:
                log(f"windows player: closed ({self.bye.get('why')}); {self.bye.get('frames', 0)} frames, "
                    f"{self.bye.get('xruns', 0)} underflows")
            if self._delegate is not None:
                self._delegate.close()
        finally:
            if got:
                self._lock.release()


# -- starting it ----------------------------------------------------------------------------------------------
class Launch:
    """Starts the Windows process on a thread (the first uv run may fetch Python and numpy), so it comes up while
    the models load; player() waits for it and returns a WinPlayer, or None after one warn."""

    def __init__(self, speaker: str = "", latency: float = WinPlayer.LATENCY_S, prepare: bool = False,
                 progress: Callable[[str], None] = lambda _: None):
        self.speaker, self.latency, self.prepare, self.progress = speaker, latency, prepare, progress
        self.proc: subprocess.Popen[bytes] | None = None
        self.error = ""
        self.done = threading.Event()
        self.abandoned = False
        self._lock = threading.Lock()

    def start(self) -> Launch:
        threading.Thread(target=self._run, name="winplayer-launch", daemon=True).start()
        return self

    def _run(self) -> None:
        try:
            paths = win_paths()
            uv = ensure_uv(paths, self.progress) if self.prepare else find_uv(paths)
            if uv is None:
                raise WinPlayerError("uv.exe is not on the PATH and not prepared yet; /live setup prepares it")
            script = stage_script(paths)
            argv, env = command(paths, uv, script, self.speaker, self.latency)
            log(f"windows player: {' '.join(argv)} (uv cache {paths.win('uv-cache')})")
            with self._lock:
                if self.abandoned:
                    return
                self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, cwd=str(paths.root), env=env)
        except Exception as e:  # noqa: BLE001 - reported by player()
            self.error = f"{type(e).__name__}: {e}" if not isinstance(e, WinPlayerError) else str(e)
        finally:
            self.done.set()

    def player(self, rate: int, reference: Any = None, guard: Any = None, timeout: float = READY_TIMEOUT_S,
               report: Callable[[str], None] = warn) -> WinPlayer | None:
        deadline = time.monotonic() + timeout
        self.done.wait(timeout)
        p = None
        if self.proc is not None:
            p = WinPlayer(self.proc)
            if p.open(rate, reference, guard, max(1.0, deadline - time.monotonic())):
                log(f"speaker: {p.describe()}")
                return p
            self.error = p.error
        self.abandon()
        if p is not None:
            p.close()
        why = self.error or f"not started within {timeout:.0f}s"
        report(f"Windows speaker unavailable ({why}); playing through WSLg instead.")
        return None

    def abandon(self) -> None:
        with self._lock:
            self.abandoned = True
            proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                proc.kill()
