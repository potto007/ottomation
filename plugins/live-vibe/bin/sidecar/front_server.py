"""The managed front server. With frontUrl empty (the default) and the llamacpp backend, /livevibe runs its own
llama.cpp llama-server: /live setup downloads a pinned prebuilt build and a default GGUF (or the frontServerBin and
frontServerModel settings name existing ones), and the front sidecar starts it on a free 127.0.0.1 port, waits for
/health, points the front at it, and stops it when live vibe stops. With frontUrl set, nothing starts here.

No orphan holds VRAM: the server runs in its own session (process group), stop() sends SIGTERM to the group, then
SIGKILL; the sidecar's quit path (Lifecycle.on_quit) and main()'s finally both call it. If the sidecar is killed
outright, Linux delivers SIGKILL to the server through PR_SET_PDEATHSIG, armed in the child and tied to the thread
that spawned it; that thread ("front-server") lives exactly as long as the server, blocked in wait(). Elsewhere
(macOS) a pidfile per server names its owning sidecar, and the next start reaps a server whose owner is gone.

GPU: the front asks gpu.admit() for its estimated need before the recognizer's claim is placed (main.run), so the
front model gets the GPU first, then the recognizer, then Kokoro. A slow front costs seconds on every reply; a
recognizer that cannot fit falls back from Kyutai to Whisper, which keeps up on the CPU; Kokoro is fine on the CPU.
Not admitted, the server runs with -ngl 0 and --device none on the CPU, with one warning that it is slow."""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from . import gpu
from .protocol import log, warn

_XDG_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
CACHE = Path(os.environ.get("LIVE_VIBE_CACHE") or _XDG_CACHE / "duplex_voice")  # speech.CACHE: Kokoro's folder too
CTX = 16384
PARALLEL = 1
HEALTH_TIMEOUT_S = 300.0  # a cold disk or a CPU-only load of a big GGUF; a missing file fails at once instead
STOP_GRACE_S = 5.0

# llama.cpp v0.5.0 is build b11146 (2026-09-23; its release lists only nightly-tag.txt, which names b11146). Recent
# enough for --jinja tool calls and response_format json_schema grammars. sha256 digests are GitHub's own, from the
# release API (assets[].digest).
LLAMA_TAG = "b11146"
LLAMA_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str
    size: int


# Linux x64 with an Nvidia GPU takes the CUDA 12.8 build plus its runtime libraries (cudart, cuBLAS): CUDA 12.8 is
# the oldest toolkit with Blackwell (sm_120) kernels, still covers pre-Turing GPUs and pre-580 drivers that CUDA 13
# drops, and matches the CUDA 12 family the rest of the sidecar loads (torch, onnxruntime-gpu). Vulkan is next where
# a Vulkan loader exists outside WSL (AMD, Intel; WSL2 has no Nvidia Vulkan driver), else the CPU build.
BUILDS: dict[str, tuple[Asset, ...]] = {
    "macos-arm64": (Asset("llama-b11146-bin-macos-arm64.tar.gz",
                          "1ad3f9eff80edb9dbef4259ad564d1720612ef7eea48fa4afed0e54f5f3d5711", 11189714),),
    "macos-x64": (Asset("llama-b11146-bin-macos-x64.tar.gz",
                        "305f0e3a17d2c01eb205cd0a62128357f1ec3b55329cb084d94e5ec0115d7a3b", 11237237),),
    "linux-x64-cuda": (Asset("llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz",
                             "c2ab9e19838513ff69d1af8d999ad717dd3c7ee4714ac04c7ed5ab9077c50e4e", 168920581),
                       Asset("cudart-llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz",
                             "1466daea60aad1144819e151b2bae19d54556cf1da6c129c4f55a5ded2637c25", 594373356)),
    "linux-x64-vulkan": (Asset("llama-b11146-bin-ubuntu-vulkan-x64.tar.gz",
                               "d3ce40fce7403cc93bcf5718fc46c6efb61ed9709f8e5d9f10c86bf0e30e8fb3", 30598492),),
    "linux-x64-cpu": (Asset("llama-b11146-bin-ubuntu-x64.tar.gz",
                            "c150306eb16b5ab696f76a8bdf810c35fd98a24e82158742e6fa28f420ff8410", 16998357),),
    "linux-arm64-cpu": (Asset("llama-b11146-bin-ubuntu-arm64.tar.gz",
                              "4aeda6fe68831547e49b7fa87607383ca5352b3d72ca5f70d52ed265f58c131f", 13598346),),
    "windows-x64-cuda": (Asset("llama-b11146-bin-win-cuda-12.4-x64.zip",
                               "3c806a6ceccc3dae1c743ceb1a1fb2cce5b76f40bfbd4c6b7b8afb6ef45a5807", 253869799),
                         Asset("cudart-llama-bin-win-cuda-12.4-x64.zip",
                               "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627)),
    "windows-x64-cpu": (Asset("llama-b11146-bin-win-cpu-x64.zip",
                              "14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1", 18560055),),
}


@dataclass(frozen=True)
class Gguf:
    repo: str
    file: str
    revision: str = "main"
    sha256: str = ""  # empty: checked against the Hub's own LFS sha256 (X-Linked-Etag) at download

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


# The front's default model: one constant. Qwen3-4B-Instruct-2507 Q4_K_M, 2.3 GiB; at -c 16384 it measured 5.3 GiB of
# VRAM with the CUDA context (4.1 at 8k, 7.6 at 32k), which estimate_gib() reproduces.
DEFAULT_MODEL = Gguf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")


class FrontServerError(Exception):
    pass


# -- settings -------------------------------------------------------------------------------------------------
@dataclass
class Settings:
    binary: str = ""  # frontServerBin: an existing llama-server; empty: the pinned download
    model: str = ""  # frontServerModel: an existing GGUF; empty: DEFAULT_MODEL, downloaded
    log: str = ""  # frontServerLog: where the server's stdout and stderr go; empty: <cache>/front-server.log
    ctx: int = CTX


def managed(args: Any) -> bool:
    """frontUrl empty and the llamacpp backend: the sidecar runs the front server itself."""
    return getattr(args, "front_backend", "llamacpp") == "llamacpp" and not str(getattr(args, "front_url", "") or "").strip()


def settings(args: Any) -> Settings:
    return Settings(str(getattr(args, "front_server_bin", "") or "").strip(),
                    str(getattr(args, "front_server_model", "") or "").strip(),
                    str(getattr(args, "front_server_log", "") or "").strip())


def build_key(system: str | None = None, machine: str | None = None, nvidia: bool | None = None,
              vulkan: bool | None = None, wsl: bool | None = None) -> str | None:
    """Which prebuilt llama.cpp fits this machine; None when none does."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    x64 = machine in ("x86_64", "amd64")
    if system == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-x64"
    if nvidia is None:
        nvidia = gpu.read_free_gib() is not None
    if system.startswith("linux"):
        if machine in ("aarch64", "arm64"):
            return "linux-arm64-cpu"
        if not x64:
            return None
        if nvidia:
            return "linux-x64-cuda"
        if wsl is None:
            wsl = "microsoft" in platform.release().lower()
        if vulkan is None:
            vulkan = ctypes.util.find_library("vulkan") is not None
        return "linux-x64-vulkan" if vulkan and not wsl else "linux-x64-cpu"
    if system == "win32" and x64:
        return "windows-x64-cuda" if nvidia else "windows-x64-cpu"
    return None


def exe_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def build_dir(key: str) -> Path:
    return CACHE / "llama.cpp" / f"{LLAMA_TAG}-{key}"


def model_file(m: Gguf = DEFAULT_MODEL) -> Path:
    return CACHE / "models" / m.repo.replace("/", "__") / m.file


def binary_path(s: Settings, key: str | None = None) -> tuple[Path, bool]:
    """(llama-server, whether it is the pinned download). The setting wins; it is never downloaded over."""
    if s.binary:
        return Path(s.binary).expanduser(), False
    key = key or build_key()
    if key is None:
        raise FrontServerError(f"no prebuilt llama.cpp for {sys.platform} {platform.machine()}; set frontServerBin "
                               "to a llama-server you built, or frontUrl to a server")
    return build_dir(key) / exe_name(), True


def model_path(s: Settings) -> tuple[Path, bool]:
    if s.model:
        return Path(s.model).expanduser(), False
    return model_file(), True


def log_path(s: Settings) -> Path:
    return Path(s.log).expanduser() if s.log else CACHE / "front-server.log"


# -- downloads ------------------------------------------------------------------------------------------------
Progress = Callable[[str], None]


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def download(url: str, dest: Path, sha256: str, size: int, progress: Progress, label: str,
             stop: threading.Event | None = None) -> Path:
    """url -> dest through dest.part (resumed with a Range request), sha256-checked before the rename. An empty sha256
    takes the Hub's X-Linked-Etag (the LFS object's sha256); with neither, the download fails."""
    import httpx

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    h = hashlib.sha256()
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(30.0, connect=15.0)) as r:
        if r.status_code not in (200, 206, 416):  # 416: the part is already whole
            raise FrontServerError(f"download {label}: HTTP {r.status_code} from {url}")
        for resp in [*r.history, r]:
            etag = resp.headers.get("x-linked-etag", "").strip('"')
            if not sha256 and re.fullmatch(r"[0-9a-f]{64}", etag):
                sha256 = etag
        if r.status_code == 200:
            have = 0
        if have:
            with open(part, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        total = size or (have + int(r.headers.get("content-length") or 0))
        got, last = have, 0.0
        if r.status_code != 416:
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    if stop is not None and stop.is_set():
                        raise FrontServerError(f"download {label}: stopped")
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    if time.monotonic() - last > 2.0:
                        last = time.monotonic()
                        pct = f" ({100 * got / total:.0f}%)" if total else ""
                        progress(f"front: downloading {label}: {_gb(got)} of {_gb(total)}{pct}")
    if not sha256:
        part.unlink(missing_ok=True)
        raise FrontServerError(f"download {label}: no sha256 to check it against")
    if h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise FrontServerError(f"download {label}: sha256 mismatch (got {h.hexdigest()[:16]}..., want {sha256[:16]}...)")
    part.replace(dest)
    return dest


def _top_folder(names: list[str]) -> str:
    """The archive's single top-level folder ("llama-b11146/"), or "" when it has none."""
    names = [n.strip("/") for n in names if n.strip("/")]
    firsts = {n.split("/", 1)[0] for n in names}
    if len(firsts) != 1:
        return ""
    top = firsts.pop()
    return f"{top}/" if any(n.startswith(f"{top}/") for n in names) else ""


def extract(archive: Path, into: Path) -> None:
    """Every archive of a build lands flat in one folder (their top folders dropped): the runtime libraries of the
    CUDA build then sit beside llama-server, where its $ORIGIN rpath and LD_LIBRARY_PATH find them."""
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            top = _top_folder(z.namelist())
            for info in z.infolist():
                name = info.filename.removeprefix(top) if top else info.filename
                if not name or name.endswith("/"):
                    continue
                target = (into / name).resolve()
                if into.resolve() not in target.parents:
                    raise FrontServerError(f"{archive.name}: {info.filename} would land outside {into}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
        return
    with tarfile.open(archive, "r:*") as t:
        members = t.getmembers()
        top = _top_folder([m.name for m in members])
        keep = []
        for m in members:
            if top:
                if not m.name.startswith(top):
                    continue  # the top folder itself
                m.name = m.name.removeprefix(top)
                if m.islnk():
                    m.linkname = m.linkname.removeprefix(top)
            keep.append(m)
        t.extractall(into, members=keep, filter="data")


def ensure_binary(s: Settings, progress: Progress, stop: threading.Event | None = None) -> Path:
    path, pinned = binary_path(s)
    if not pinned:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FrontServerError(f"frontServerBin {path} is not an executable file")
        return path
    if path.is_file():
        return path
    key = path.parent.name.removeprefix(f"{LLAMA_TAG}-")
    folder = path.parent
    downloads = CACHE / "llama.cpp" / "downloads"
    for a in BUILDS[key]:
        got = download(LLAMA_RELEASE + a.name, downloads / a.name, a.sha256, a.size, progress, a.name, stop)
        progress(f"front: unpacking {a.name}")
        extract(got, folder)
    if not path.is_file():
        raise FrontServerError(f"{BUILDS[key][0].name} has no {exe_name()}")
    for a in BUILDS[key]:
        (downloads / a.name).unlink(missing_ok=True)  # unpacked and checked: the archives are dead weight
    return path


def ensure_model(s: Settings, progress: Progress, stop: threading.Event | None = None) -> Path:
    path, default = model_path(s)
    if not default:
        if not path.is_file():
            raise FrontServerError(f"frontServerModel {path} is not a file")
        return path
    m = DEFAULT_MODEL
    return download(m.url, path, m.sha256, 0, progress, m.file, stop)


def missing(s: Settings) -> list[str]:
    """What a start would have to download first (empty: offline is fine)."""
    out = []
    b, pinned = binary_path(s)
    if pinned and not b.is_file():
        out.append(f"llama.cpp {LLAMA_TAG} ({b.parent.name.removeprefix(LLAMA_TAG + '-')})")
    m, default = model_path(s)
    if default and not m.is_file():
        out.append(m.name)
    return out


# -- GPU need -------------------------------------------------------------------------------------------------
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
OVERHEAD_GIB = 0.75  # CUDA context and compute buffers at -np 1: fits the measured 4.1 / 5.3 / 7.6 GiB at 8k/16k/32k


def gguf_meta(path: Path, suffixes: set[str]) -> dict[str, Any]:
    """general.architecture and the "<arch>.<suffix>" keys of a GGUF (v2/v3). The model's hyperparameters come before
    the tokenizer's big arrays, so reading stops at the first tokenizer key."""
    out: dict[str, Any] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"GGUF v{version} is too old")
        _tensors, count = struct.unpack("<QQ", f.read(16))

        def string() -> str:
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(t: int) -> Any:
            if t in _GGUF_SCALAR:
                fmt = _GGUF_SCALAR[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return string()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                return [value(et) for _ in range(n)]
            raise ValueError(f"GGUF value type {t}")

        for _ in range(count):
            key = string()
            if key.startswith("tokenizer.") and "general.architecture" in out:
                break
            t, = struct.unpack("<I", f.read(4))
            v = value(t)
            if key == "general.architecture" or key.split(".", 1)[-1] in suffixes:
                out[key] = v
    return out


_KV_KEYS = {"block_count", "embedding_length", "attention.head_count",
            "attention.head_count_kv", "attention.key_length", "attention.value_length"}


def kv_bytes_per_token(path: Path) -> int | None:
    """f16 K and V for every attention layer; None when the GGUF does not say."""
    try:
        meta = gguf_meta(path, _KV_KEYS)
        arch = meta["general.architecture"]
        g = lambda k: meta.get(f"{arch}.{k}")  # noqa: E731
        layers, heads = int(g("block_count")), g("attention.head_count")
        heads = max(heads) if isinstance(heads, list) else int(heads)
        kv = g("attention.head_count_kv") or heads
        kv_heads = sum(int(x) for x in kv) if isinstance(kv, list) else layers * int(kv)
        k_len = int(g("attention.key_length") or int(g("embedding_length")) // heads)
        v_len = int(g("attention.value_length") or k_len)
        return kv_heads * (k_len + v_len) * 2
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, struct.error):
        return None


def estimate_gib(path: Path, ctx: int) -> float:
    """Weights + KV cache at `ctx` + OVERHEAD_GIB. The default model at 16k: 2.33 + 2.25 + 0.75 = 5.3 GiB."""
    weights = path.stat().st_size / 2**30 if path.exists() else 2.5
    per_token = kv_bytes_per_token(path)
    if per_token is None:
        return max(weights + OVERHEAD_GIB, gpu.NEED_GIB["front"] * ctx / CTX)
    return weights + per_token * ctx / 2**30 + OVERHEAD_GIB


def placement(model: Path, ctx: int) -> tuple[bool, str]:
    """(on the GPU, why). Apple silicon is unified memory: Metal, no budget. Elsewhere the shared budget decides, and
    an admitted front holds its claim ("front" in gpu.py) until its memory shows in the reading (server healthy)."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return True, "Metal"
    need = estimate_gib(model, ctx)
    why = gpu.admit("front", need)
    if why is None:
        return True, f"about {need:.1f} GiB"
    return False, why


# -- the process ----------------------------------------------------------------------------------------------
def free_port(host: str = "127.0.0.1") -> int:
    """An ephemeral port the kernel just handed out (so never a fixed one another server, such as :8080, uses)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def command(binary: Path, model: Path, port: int, ctx: int, on_gpu: bool) -> list[str]:
    cmd = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "--jinja",
           "-c", str(ctx), "-np", str(PARALLEL), "--no-webui"]
    return cmd + (["-ngl", "99"] if on_gpu else ["-ngl", "0", "--device", "none"])


_PR_SET_PDEATHSIG = 1


def _death_signal() -> Callable[[], None] | None:
    """Linux: a preexec_fn that arms SIGKILL for when the spawning thread dies (the sidecar crashing or being killed
    takes that thread with it). The libc symbol is resolved here, before the fork."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()

    def arm() -> None:
        prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        if os.getppid() != parent:  # the parent died before the signal was armed
            os.kill(os.getpid(), signal.SIGKILL)

    return arm


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def pid_dir() -> Path:
    return CACHE / "front-server"


def reap_stale(folder: Path | None = None) -> list[int]:
    """Servers a dead sidecar left behind (macOS, or a kill that beat every cleanup): its pidfile names the owner, and
    a server is killed only while its command line still matches what the file recorded."""
    folder = folder or pid_dir()
    reaped = []
    for f in folder.glob("*.pid") if folder.is_dir() else []:
        try:
            rec = json.loads(f.read_text())
            pid, owner, port, binary = int(rec["pid"]), int(rec["owner"]), int(rec["port"]), str(rec["binary"])
        except (OSError, ValueError, KeyError, TypeError):
            f.unlink(missing_ok=True)
            continue
        if _alive(owner):  # its sidecar (or this one) still runs it
            continue
        line = _cmdline(pid) if _alive(pid) else ""
        if line and Path(binary).name in line and f"--port {port}" in line:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            reaped.append(pid)
            log(f"front: stopped llama-server pid {pid} on :{port}, left behind by sidecar pid {owner}")
        f.unlink(missing_ok=True)
    return reaped


_running: list[FrontServer] = []
_running_lock = threading.Lock()


class FrontServer:
    """One llama-server. start() picks the port (so .url is known at once) and, when the files are already here,
    the GPU placement, then a "front-server" thread downloads what is missing, spawns the server, waits for /health
    and then stays in wait() for as long as the server lives."""

    def __init__(self, s: Settings, progress: Progress | None = None, health_timeout: float = HEALTH_TIMEOUT_S):
        self.s = s
        self.progress = progress or log
        self.health_timeout = health_timeout
        self.log_file = log_path(s)
        self.port = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self.on_gpu: bool | None = None
        self.where = ""
        self.cmd: list[str] = []
        self.binary: Path | None = None
        self.model: Path | None = None
        self.error = ""
        # llama-server answers any origin (CORS *): a per-start key keeps web pages and other local users out. It
        # goes in LLAMA_API_KEY, so it is never on the command line (ps) or in the log; /health needs none.
        self.api_key = secrets.token_urlsafe(24)
        self.ready_s = 0.0
        self.offloaded = ""
        self._ready = threading.Event()
        self._done = threading.Event()  # ready or failed
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pidfile: Path | None = None
        self._log_start = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FrontServer:
        reap_stale()
        self.port = free_port()
        b, _ = binary_path(self.s)
        m, _ = model_path(self.s)
        if b.is_file() and m.is_file():
            self.on_gpu, self.where = placement(m, self.s.ctx)  # now, before the recognizer's claim
        with _running_lock:
            _running.append(self)
        self._thread = threading.Thread(target=self._run, name="front-server", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        t0 = time.monotonic()
        try:
            if gaps := missing(self.s):
                warn(f"front: downloading {', '.join(gaps)} first (/live setup does this ahead of time); until it "
                     "is ready, speech goes straight to Claude")
            self.binary = ensure_binary(self.s, self.progress, self._stopping)
            self.model = ensure_model(self.s, self.progress, self._stopping)
            if self.on_gpu is None:
                self.on_gpu, self.where = placement(self.model, self.s.ctx)
            if not self.on_gpu:
                warn(f"front: running the front model on the CPU, which is slow (seconds per reply): {self.where}")
            self._spawn()
            self._wait_health(t0)
            self._ready.set()
            self._done.set()
            assert self.proc is not None
            code = self.proc.wait()
            if not self._stopping.is_set():
                warn(f"front: llama-server exited (code {code}); speech goes straight to Claude. Log: {self.log_file}")
        except Exception as e:  # noqa: BLE001 - the front is optional; the voice runs on without it
            self.error = str(e) if isinstance(e, FrontServerError) else f"{type(e).__name__}: {e}"
            if not self._stopping.is_set():
                warn(f"front: managed llama-server failed: {self.error[:400]}")
        finally:
            gpu.release("front")
            self._done.set()
            self._kill()  # this thread is the one PR_SET_PDEATHSIG watches: the server never outlives it

    def _spawn(self) -> None:
        assert self.binary is not None and self.model is not None
        self.cmd = command(self.binary, self.model, self.port, self.s.ctx, bool(self.on_gpu))
        env = {**os.environ, "LLAMA_API_KEY": self.api_key}
        _, pinned = binary_path(self.s)
        if pinned and sys.platform.startswith("linux"):  # the CUDA runtime libraries unpacked beside the binary
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(self.binary.parent), env.get("LD_LIBRARY_PATH")]))
        if not self.on_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""  # no CUDA context either: it would cost VRAM the budget refused
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        out: IO[bytes] = open(self.log_file, "ab")
        try:
            self._log_start = out.tell()
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            out.write((f"--- {stamp} live-vibe front server, sidecar pid {os.getpid()}: port {self.port}, model "
                       f"{self.model}, {'GPU' if self.on_gpu else 'CPU'} ({self.where})\n"
                       f"--- {shlex.join(self.cmd)}\n").encode())
            out.flush()
            kw: dict[str, Any] = {}
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                kw["start_new_session"] = True
                if (arm := _death_signal()) is not None:
                    kw["preexec_fn"] = arm
            with self._lock:
                if self._stopping.is_set():
                    raise FrontServerError("stopped before it started")
                self.proc = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                             env=env, **kw)
        finally:
            out.close()
        self._write_pidfile()
        log(f"front: managed llama-server pid {self.proc.pid} on {self.url}, model {self.model.name}, "
            f"{'GPU' if self.on_gpu else 'CPU'} ({self.where}); log {self.log_file}")
        log(f"front: {shlex.join(self.cmd)}")

    def _write_pidfile(self) -> None:
        assert self.proc is not None and self.binary is not None
        try:
            pid_dir().mkdir(parents=True, exist_ok=True)
            self._pidfile = pid_dir() / f"{self.proc.pid}.pid"
            self._pidfile.write_text(json.dumps({"pid": self.proc.pid, "owner": os.getpid(), "port": self.port,
                                                 "binary": str(self.binary)}))
        except OSError:
            self._pidfile = None

    def log_tail(self, lines: int = 6) -> str:
        try:
            with open(self.log_file, "rb") as f:
                f.seek(self._log_start)
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        rows = [x.strip() for x in text.splitlines() if x.strip() and not x.startswith("--- ")]
        return " | ".join(rows[-lines:])

    def _wait_health(self, t0: float) -> None:
        deadline = time.monotonic() + self.health_timeout
        while True:
            if self._stopping.is_set():
                raise FrontServerError("stopped while loading")
            assert self.proc is not None
            if (code := self.proc.poll()) is not None:
                raise FrontServerError(f"llama-server exited with code {code} while loading: {self.log_tail()} "
                                       f"(log {self.log_file})")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass  # refused while it binds, 503 while it loads
            if time.monotonic() > deadline:
                raise FrontServerError(f"llama-server not healthy after {self.health_timeout:.0f}s: {self.log_tail()}")
            time.sleep(0.25)
        self.ready_s = time.monotonic() - t0
        gpu.release("front")  # its memory shows in free-VRAM readings from here on
        self.offloaded = self._offload_line()
        if self.on_gpu and re.search(r"offloaded 0/\d+ layers", self.offloaded):
            self.on_gpu = False
            warn(f"front: llama-server put no layers on the GPU ({self.offloaded}); the front runs on the CPU and "
                 f"is slow. Log: {self.log_file}")
        log(f"front: llama-server ready in {self.ready_s:.1f}s on {self.url}"
            + (f" ({self.offloaded})" if self.offloaded else ""))

    def _offload_line(self) -> str:
        try:
            with open(self.log_file, "rb") as f:
                f.seek(self._log_start)
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        m = re.search(r"offloaded (\d+/\d+) layers to GPU", text)
        return f"offloaded {m.group(1)} layers to GPU" if m else ""

    def wait_ready(self, timeout: float | None = None) -> bool:
        """True once /health answers; False when it failed, stopped, or `timeout` passed."""
        self._done.wait(timeout)
        return self._ready.is_set() and not self._stopping.is_set()

    def terminate(self) -> None:
        """SIGTERM to the server's group, without waiting: for the quit path, which must not block."""
        self._stopping.set()
        with self._lock:
            p = self.proc
        if p is not None and p.poll() is None:
            _signal_group(p, signal.SIGTERM)

    def _kill(self, grace: float = STOP_GRACE_S) -> None:
        with self._lock:
            p = self.proc
        if p is not None and p.poll() is None:
            _signal_group(p, signal.SIGTERM)
            try:
                p.wait(grace)
            except subprocess.TimeoutExpired:
                _signal_group(p, signal.SIGKILL)
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    p.wait(2)
        if self._pidfile is not None:
            self._pidfile.unlink(missing_ok=True)

    def stop(self, grace: float = STOP_GRACE_S) -> None:
        """Ends the server and waits for it (SIGTERM, then SIGKILL after `grace`). Idempotent."""
        self._stopping.set()
        self._kill(grace)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(grace + 3)
        gpu.release("front")
        with _running_lock:
            if self in _running:
                _running.remove(self)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None


def _signal_group(p: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    with contextlib.suppress(OSError):
        if sys.platform == "win32":
            p.kill() if sig == signal.SIGKILL else p.terminate()
        else:
            os.killpg(p.pid, sig)  # its own session: the group id is its pid


# -- the sidecar's entry points -------------------------------------------------------------------------------
def start_managed(args: Any, on_quit: Callable[[Callable[[], None]], None] | None = None) -> FrontServer | None:
    """/livevibe with frontUrl empty: start the server and point args.front_url at it. None otherwise, or when it
    cannot even begin (a warning says why; speech then goes straight to Claude)."""
    if getattr(args, "mode", "front") != "front" or not managed(args):
        return None
    try:
        server = FrontServer(settings(args)).start()
    except FrontServerError as e:
        warn(f"front: {e}")
        return None
    args.front_url = server.url
    if on_quit is not None:
        on_quit(server.terminate)
    return server


def stop_all() -> None:
    with _running_lock:
        servers = list(_running)
    for s in servers:
        s.stop()


def authorize(brain: Any) -> None:
    """The managed server's key on every request the front brain makes (its httpx client's default headers)."""
    with _running_lock:
        server = _running[-1] if _running else None
    headers = getattr(getattr(brain, "client", None), "headers", None)
    if server is not None and headers is not None:
        headers["Authorization"] = f"Bearer {server.api_key}"


async def when_ready(warm: Awaitable[Any]) -> None:
    """Runs `warm` (the front's warm-up) once the managed server answers /health; right away without one."""
    with _running_lock:
        server = _running[-1] if _running else None
    if server is not None and not await asyncio.to_thread(server.wait_ready):
        if hasattr(warm, "close"):
            warm.close()  # type: ignore[attr-defined]
        return
    await warm


def version(binary: Path) -> str:
    """What --version says, as parse_version() reads it."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(binary.parent), env.get("LD_LIBRARY_PATH")]))
    try:
        r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return f"(--version failed: {type(e).__name__})"
    return parse_version(r.stdout + r.stderr)


def parse_version(text: str) -> str:
    """'version: 0.5.0-dev (build 11146, commit 7fe450e19)' -> 'b11146 (0.5.0-dev)'; older builds print
    'version: 6123 (abc1234)' -> 'b6123 (abc1234)'."""
    if m := re.search(r"version:\s*(\S+)\s*\(build (\d+), commit ([0-9a-f]+)\)", text):
        return f"b{m.group(2)} ({m.group(1)}, {m.group(3)})"
    if m := re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text):
        return f"b{m.group(1)} ({m.group(2)})"
    line = next((x.strip() for x in text.splitlines() if x.strip().startswith("version:")), "")
    return line.removeprefix("version:").strip()[:80] or "(unknown version)"
