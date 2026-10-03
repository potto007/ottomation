"""--unit checks for the managed front server (front_server.py): mode and settings, the build choice, the port, the
command line, GPU placement and its CPU fallback, the GGUF estimate, downloads and unpacking, and the process life
(health wait, timeout, a crash while loading, stop, SIGKILL after the grace, no orphan when the sidecar dies) against a
fake llama-server: a few lines of Python that answer /health the way the real one does."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

from . import front_server as fs
from . import gpu, protocol

FAKE = r'''import json, os, signal, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
mode = os.environ.get("FAKE_MODE", "ok")
print("fake llama-server", " ".join(args), "CUDA_VISIBLE_DEVICES=%r" % os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
if mode == "crash":
    print("error: failed to load model", flush=True)
    sys.exit(3)
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
gpu = "-ngl" in args and args[args.index("-ngl") + 1] != "0"
print("load_tensors: offloaded %s layers to GPU" % ("37/37" if gpu else "0/37"), flush=True)
polls = [0]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        polls[0] += 1
        ok = self.path == "/health" and mode != "never" and polls[0] > 2
        self.send_response(200 if ok else 503)
        self.send_header("Content-Length", "0")
        self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.headers.get("Authorization") != "Bearer " + os.environ.get("LLAMA_API_KEY", ""):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), H).serve_forever()
'''


def gguf_bytes(arch: str = "qwen3", layers: int = 36, heads: int = 32, kv: int = 8, head_dim: int = 128) -> bytes:
    """A GGUF v3 header with the hyperparameters the estimate reads, then a tokenizer array it must not read."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [(s("general.architecture") + struct.pack("<I", 8) + s(arch)),
           (s(f"{arch}.block_count") + struct.pack("<II", 4, layers)),
           (s(f"{arch}.embedding_length") + struct.pack("<II", 4, heads * head_dim)),
           (s(f"{arch}.attention.head_count") + struct.pack("<II", 4, heads)),
           (s(f"{arch}.attention.head_count_kv") + struct.pack("<II", 4, kv)),
           (s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("b"))]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs)


@contextmanager
def patched(obj: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextmanager
def captured() -> Iterator[list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    protocol.capture(out)
    try:
        yield out
    finally:
        protocol.capture(None)


def gone(pid: int, within: float = 3.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if Path(f"/proc/{pid}/stat").exists():  # a zombie whose parent has not reaped it yet counts as gone
            try:
                if Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z"):
                    return True
            except (OSError, IndexError):
                pass
        time.sleep(0.05)
    return False


def front_server_units(check: Callable[[Any, str], bool]) -> None:
    with tempfile.TemporaryDirectory() as d, patched(fs, "CACHE", Path(d) / "cache"):
        root = Path(d)
        settings_units(check, root)
        gguf_units(check, root)
        placement_units(check)
        download_units(check, root)
        if sys.platform != "win32":
            process_units(check, root)


def settings_units(check: Callable[[Any, str], bool], root: Path) -> None:
    def ns(**kw: Any) -> argparse.Namespace:
        return argparse.Namespace(**{"mode": "front", "front_backend": "llamacpp", "front_url": "", **kw})

    check(fs.managed(ns()) and not fs.managed(ns(front_url="http://127.0.0.1:8080"))
          and not fs.managed(ns(front_backend="anthropic")) and fs.managed(ns(front_url="  ")),
          "managed front: only the llamacpp backend with frontUrl empty")
    a = ns(front_url="http://gpu-box:8080", front_server_bin="/nope/llama-server")
    check(fs.start_managed(a) is None and a.front_url == "http://gpu-box:8080" and not fs._running,
          "managed front: frontUrl set starts nothing and keeps the URL, whatever the server settings say")
    check(fs.start_managed(ns(mode="live")) is None, "managed front: /live never starts it")

    s = fs.settings(ns(front_server_bin="~/bin/llama-server", front_server_model="/m/x.gguf", front_server_log="/tmp/l.log"))
    check(fs.binary_path(s) == (Path.home() / "bin/llama-server", False) and fs.model_path(s) == (Path("/m/x.gguf"), False)
          and fs.log_path(s) == Path("/tmp/l.log"), "settings: frontServerBin, -Model and -Log win over the defaults")
    s = fs.settings(ns())
    b, pinned = fs.binary_path(s, "linux-x64-cuda")
    check(pinned and b == fs.CACHE / "llama.cpp" / f"{fs.LLAMA_TAG}-linux-x64-cuda" / "llama-server"
          and fs.model_path(s) == (fs.CACHE / "models" / "unsloth__Qwen3-4B-Instruct-2507-GGUF"
                                   / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", True)
          and fs.log_path(s) == fs.CACHE / "front-server.log", f"settings: empty means the pinned download {b}")
    try:
        fs.ensure_binary(fs.Settings(binary=str(root / "missing")), lambda _: None)
        msg = ""
    except fs.FrontServerError as e:
        msg = str(e)
    check("not an executable" in msg, f"settings: a frontServerBin that is not there fails clearly ({msg!r})")

    keys = (fs.build_key("darwin", "arm64"), fs.build_key("linux", "x86_64", nvidia=True),
            fs.build_key("linux", "x86_64", nvidia=False, vulkan=True, wsl=False),
            fs.build_key("linux", "x86_64", nvidia=False, vulkan=True, wsl=True),
            fs.build_key("linux", "x86_64", nvidia=False, vulkan=False, wsl=False), fs.build_key("linux", "aarch64"),
            fs.build_key("win32", "AMD64", nvidia=True), fs.build_key("linux", "riscv64", nvidia=False))
    check(keys == ("macos-arm64", "linux-x64-cuda", "linux-x64-vulkan", "linux-x64-cpu", "linux-x64-cpu",
                   "linux-arm64-cpu", "windows-x64-cuda", None) and all(k in fs.BUILDS for k in keys if k),
          f"build choice: Metal, CUDA with an Nvidia GPU, Vulkan off WSL, else CPU {keys}")
    check(all(len(a.sha256) == 64 and a.size > 0 and fs.LLAMA_TAG in a.name or a.name.startswith("cudart-llama-bin")
              for assets in fs.BUILDS.values() for a in assets), "builds: every asset pinned with a sha256 and a size")

    v = (fs.parse_version("0.00.000.319 I srv  llama_server: initializing ...\n"
                          "version: 0.5.0-dev (build 11146, commit 7fe450e19)"),
         fs.parse_version("version: 741 (afeebe103)\nbuilt with GNU 13.3.0"), fs.parse_version("garbage"))
    check(v == ("b11146 (0.5.0-dev, 7fe450e19)", "b741 (afeebe103)", "(unknown version)"), f"version: both formats {v}")
    ports = {fs.free_port() for _ in range(5)}
    ok = True
    for p in ports:
        with socket.socket() as sk:
            try:
                sk.bind(("127.0.0.1", p))
            except OSError:
                ok = False
    check(ok and all(1024 < p < 65536 and p not in (8080, 8091) for p in ports),
          f"port: an ephemeral 127.0.0.1 port, free to bind {sorted(ports)}")
    on = fs.command(Path("/b/llama-server"), Path("/m.gguf"), 4000, fs.CTX, True)
    off = fs.command(Path("/b/llama-server"), Path("/m.gguf"), 4000, fs.CTX, False)
    joined = " ".join(on)
    check("--jinja" in on and "-c 16384" in joined and "-np 1" in joined and "--host 127.0.0.1 --port 4000" in joined
          and on[-2:] == ["-ngl", "99"] and off[-4:] == ["-ngl", "0", "--device", "none"],
          f"command: --jinja, -c 16384, -np 1, loopback; -ngl 99 on the GPU, -ngl 0 --device none on the CPU {joined}")


def gguf_units(check: Callable[[Any, str], bool], root: Path) -> None:
    g = root / "tiny.gguf"
    g.write_bytes(gguf_bytes())
    per = fs.kv_bytes_per_token(g)
    est = fs.estimate_gib(g, 16384)
    check(per == 36 * 8 * 256 * 2 and abs(est - (g.stat().st_size / 2**30 + 2.25 + fs.OVERHEAD_GIB)) < 1e-6,
          f"gguf: f16 KV per token from the header (Qwen3-4B: {per} bytes, 2.25 GiB at 16k)")
    bad = root / "bad.gguf"
    bad.write_bytes(b"nope")
    check(fs.kv_bytes_per_token(bad) is None and fs.estimate_gib(bad, 16384) >= gpu.NEED_GIB["front"],
          "gguf: an unreadable header falls back to the measured default need")


def placement_units(check: Callable[[Any, str], bool]) -> None:
    if sys.platform == "darwin":
        return
    g = Path(tempfile.mkstemp(suffix=".gguf")[1])
    try:
        g.write_bytes(gguf_bytes())
        with patched(gpu, "read_free_gib", lambda: 20.0):
            on, why = fs.placement(g, 16384)
            held = gpu.claimed()
        gpu.release("front")
        with patched(gpu, "read_free_gib", lambda: 4.0):
            off, why_off = fs.placement(g, 16384)
        with patched(gpu, "read_free_gib", lambda: None):
            none, why_none = fs.placement(g, 16384)
        with patched(gpu, "read_free_gib", lambda: 7.0):
            first, _ = fs.placement(g, 16384)  # main.run asks for the front before the recognizer's placeholder
        gpu.release("front")
        gpu.hold("stt", 3.2)
        with patched(gpu, "read_free_gib", lambda: 7.0):
            after, _ = fs.placement(g, 16384)  # asked after the recognizer's 3.2 GiB claim, it would not fit
        gpu.release("stt")
        gpu.release("front")
    finally:
        g.unlink()
    check(on and held > 2.9 and not off and "leaves 3.0 GiB free" in why_off and not none and "unknown" in why_none,
          f"placement: GPU when it fits (claim {held:.1f} GiB held), CPU when it does not ({why_off!r}) or is unknown")
    check(first and not after and not gpu.claimed(), "placement: the front's claim comes first; asked second it loses")


class _Files(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}

    def log_message(self, *_: Any) -> None:
        pass

    def do_GET(self) -> None:
        data = self.files.get(self.path)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        if self.path.endswith(".gguf"):
            self.send_header("X-Linked-Etag", f'"{hashlib.sha256(data).hexdigest()}"')
        self.end_headers()
        self.wfile.write(body)


def tgz(entries: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        dirs = {n.split("/")[0] for n in entries}
        for dname in dirs:
            info = tarfile.TarInfo(dname)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            t.addfile(info)
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            t.addfile(info)
    return buf.getvalue()


def download_units(check: Callable[[Any, str], bool], root: Path) -> None:
    payload = os.urandom(300_000)
    main_tgz = tgz({"llama-b1/llama-server": b"#!/bin/sh\n", "llama-b1/libllama.so.0": b"x"},
                   {"llama-b1/libllama.so": "libllama.so.0"})
    cudart_tgz = tgz({"cudart-llama-b1/libcudart.so.12": b"y"})
    _Files.files = {"/m.gguf": payload, "/a.tar.gz": main_tgz, "/c.tar.gz": cudart_tgz}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    seen: list[str] = []
    try:
        dest = root / "dl" / "m.gguf"
        part = dest.with_name(dest.name + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(payload[:100_000])  # a download cut off earlier
        fs.download(f"{base}/m.gguf", dest, "", 0, seen.append, "m.gguf")  # sha256 from X-Linked-Etag, resumed
        check(dest.read_bytes() == payload and not part.exists(), "download: resumes a .part and checks the Hub's sha256")
        bad = root / "dl" / "bad.gguf"
        try:
            fs.download(f"{base}/m.gguf", bad, "0" * 64, 0, seen.append, "bad")
            msg = ""
        except fs.FrontServerError as e:
            msg = str(e)
        check("mismatch" in msg and not bad.exists() and not bad.with_name("bad.gguf.part").exists(),
              f"download: a sha256 mismatch keeps nothing ({msg!r})")
        into = root / "build"
        for name, data in (("a.tar.gz", main_tgz), ("c.tar.gz", cudart_tgz)):
            got = fs.download(f"{base}/{name}", root / "dl" / name, hashlib.sha256(data).hexdigest(), len(data),
                              seen.append, name)
            fs.extract(got, into)
        names = sorted(p.name for p in into.iterdir())
        check(names == ["libcudart.so.12", "libllama.so", "libllama.so.0", "llama-server"]
              and os.access(into / "llama-server", os.X_OK) and (into / "libllama.so").is_symlink(),
              f"unpack: both archives land flat beside llama-server, modes and links kept {names}")
    finally:
        srv.shutdown()
        srv.server_close()


def fake_binary(root: Path) -> Path:
    b = root / "fake-llama-server"
    b.write_text(f"#!{sys.executable}\n{FAKE}")
    b.chmod(0o755)
    return b


def process_units(check: Callable[[Any, str], bool], root: Path) -> None:
    binary, model = fake_binary(root), root / "model.gguf"
    model.write_bytes(gguf_bytes())
    logf = root / "front.log"
    s = fs.Settings(str(binary), str(model), str(logf))

    def run(mode: str, free: float | None, timeout: float = 30.0) -> fs.FrontServer:
        os.environ["FAKE_MODE"] = mode
        with patched(gpu, "read_free_gib", lambda: free):
            return fs.FrontServer(s, health_timeout=timeout).start()

    try:
        with captured() as out:
            srv = run("ok", 40.0)
            ready = srv.wait_ready(20)
            pid = srv.pid
            pidfiles = list((fs.pid_dir()).glob("*.pid"))
            with urllib.request.urlopen(f"{srv.url}/health", timeout=5) as r:
                health = r.status
            brain = types.SimpleNamespace(client=types.SimpleNamespace(headers={}))
            fs.authorize(brain)
            codes = []
            for headers in ({}, brain.client.headers):
                req = urllib.request.Request(f"{srv.url}/v1/chat/completions", data=b"{}", headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        codes.append(r.status)
                except urllib.error.HTTPError as e:
                    codes.append(e.code)
            srv.stop()
        text = logf.read_text()
        check(ready and health == 200 and srv.on_gpu and "offloaded 37/37" in srv.offloaded
              and f"--port {srv.port}" in text and "--- " in text and "GPU" in text and len(pidfiles) == 1,
              f"start: health waited out, command line and port in the log, a pidfile ({srv.ready_s:.1f}s)")
        check(codes == [401, 200] and srv.api_key not in text and srv.api_key not in " ".join(srv.cmd),
              f"api key: the server takes requests only with the per-start key, which stays out of argv and log {codes}")
        check(pid is not None and gone(pid) and not list(fs.pid_dir().glob("*.pid")) and not gpu.claimed()
              and not fs._running, "stop: the process is gone, its pidfile and GPU claim with it")
        check(any("managed llama-server pid" in str(m.get("text")) for m in out), "start: logs pid, url, model, place")

        with captured() as out:
            srv = run("ok", 2.0)
            ready = srv.wait_ready(20)
            srv.stop()
        warned = [str(m["text"]) for m in out if m.get("type") == "warn"]
        check(ready and srv.on_gpu is False and "--device none" in " ".join(srv.cmd)
              and "CUDA_VISIBLE_DEVICES=''" in logf.read_text() and len(warned) == 1 and "slow" in warned[0],
              f"cpu fallback: not admitted, -ngl 0 --device none, no CUDA device, one warning {warned}")

        with captured():
            srv = run("never", 40.0, timeout=1.5)
            ready = srv.wait_ready(20)
            pid = srv.pid
            srv.stop()
        check(not ready and "not healthy after" in srv.error and pid is not None and gone(pid) and not gpu.claimed(),
              f"timeout: no /health in time is an error and the process is stopped ({srv.error[:60]!r})")

        with captured() as out:
            srv = run("crash", 40.0)
            ready = srv.wait_ready(20)
            srv.stop()
        check(not ready and "exited with code 3" in srv.error and "failed to load model" in srv.error,
              f"crash while loading: the exit code and the log's tail ({srv.error[:90]!r})")

        with captured():
            srv = run("ignore_term", 40.0)
            srv.wait_ready(20)
            pid = srv.pid
            t = time.monotonic()
            srv.stop(grace=0.5)
        check(pid is not None and gone(pid) and time.monotonic() - t < 5,
              "stop: SIGKILL after the grace when SIGTERM is ignored")

        async def warm() -> str:
            return "warmed"

        with captured():
            srv = run("crash", 40.0)
            coro = warm()
            asyncio.run(fs.when_ready(coro))
            srv.stop()
        check(coro.cr_frame is None, "when_ready: a failed server never runs the warm-up (and closes it)")
        if sys.platform.startswith("linux"):
            orphan_units(check, root, binary, model)
        reap_units(check, root, binary)
    finally:
        os.environ.pop("FAKE_MODE", None)


def orphan_units(check: Callable[[Any, str], bool], root: Path, binary: Path, model: Path) -> None:
    """A sidecar killed with SIGKILL: no cleanup runs, PR_SET_PDEATHSIG takes the server down with it."""
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); from sidecar import front_server as fs, gpu; "
            "gpu.read_free_gib = lambda: 40.0; "
            "srv = fs.FrontServer(fs.Settings(sys.argv[2], sys.argv[3], sys.argv[4])).start(); "
            "print(srv.pid if srv.wait_ready(20) else -1, flush=True); time.sleep(60)")
    env = {**os.environ, "FAKE_MODE": "ok", "LIVE_VIBE_CACHE": str(fs.CACHE)}
    child = subprocess.Popen([sys.executable, "-c", code, str(Path(fs.__file__).parent.parent), str(binary), str(model),
                              str(root / "orphan.log")], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout is not None
        pid = -1
        for line in child.stdout:  # protocol lines first, then the pid
            if line.strip().lstrip("-").isdigit():
                pid = int(line)
                break
        child.send_signal(signal.SIGKILL)
        child.wait(5)
        check(pid > 0 and gone(pid, 5.0), f"no orphan: SIGKILL to the sidecar takes llama-server (pid {pid}) with it")
    finally:
        if child.poll() is None:
            child.kill()


def reap_units(check: Callable[[Any, str], bool], root: Path, binary: Path) -> None:
    """A server whose owner died is stopped at the next start; one whose owner lives is left alone."""
    dead = subprocess.Popen(["true"])
    dead.wait()
    port = fs.free_port()
    p = subprocess.Popen([str(binary), "--port", str(port)], env={**os.environ, "FAKE_MODE": "never"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    live = subprocess.Popen([str(binary), "--port", str(fs.free_port())], env={**os.environ, "FAKE_MODE": "never"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        folder = fs.pid_dir()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{p.pid}.pid").write_text(json.dumps({"pid": p.pid, "owner": dead.pid, "port": port,
                                                         "binary": str(binary)}))
        (folder / f"{live.pid}.pid").write_text(json.dumps({"pid": live.pid, "owner": os.getppid(), "port": 1,
                                                            "binary": str(binary)}))
        time.sleep(0.3)  # the fakes are up
        with captured():
            reaped = fs.reap_stale()
        p.wait(5)
        check(reaped == [p.pid] and live.poll() is None and not (folder / f"{p.pid}.pid").exists()
              and (folder / f"{live.pid}.pid").exists(),
              f"reap: a dead owner's server is stopped, a live owner's kept {reaped}")
    finally:
        for proc in (p, live):
            if proc.poll() is None:
                proc.kill()
                proc.wait(5)
        for f in fs.pid_dir().glob("*.pid"):
            f.unlink()
