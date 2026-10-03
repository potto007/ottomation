"""The sidecar's contract with the mod (hooks/register.tsx): JSON lines on stdout, a token-guarded HTTP control
server on 127.0.0.1, and the process lifecycle (signals, parent death, a hard deadline on shutdown)."""
from __future__ import annotations

import datetime
import hmac
import json
import logging
import logging.handlers
import os
import platform
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any, Callable

TOKEN_HEADER = "X-Live-Token"
MAX_BODY = 256 * 1024  # a /speak or /event body; Claude's longest answers are far below this
QUIT_GRACE_S = 5.0  # after a quit request, the process ends within this even if a thread hangs (MLX, PortAudio)

_out: IO[str] | None = None
_out_lock = threading.RLock()  # RLock: a signal handler on the main thread may emit while it holds the lock
_capture: list[dict[str, Any]] | None = None  # the self-test reads emits here
_on_broken: Callable[[str], None] = lambda _: None


def encode(obj: dict[str, Any]) -> str:
    """One protocol line. json.dumps escapes newlines inside strings, so a message is always exactly one line."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def claim_stdout() -> None:
    """Keep stdout for protocol lines only: the real fd 1 moves to a private stream, and fd 1 and sys.stdout now
    point at stderr, so a library's print or C-level write (espeak, ctranslate2, MLX) lands in the mod's debug log
    instead of breaking its JSON parser."""
    global _out
    sys.stdout.flush()
    _out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def capture(into: list[dict[str, Any]] | None) -> None:
    global _capture
    _capture = into


def on_broken_stdout(fn: Callable[[str], None]) -> None:
    global _on_broken
    _on_broken = fn


def emit(**obj: Any) -> None:
    if _capture is not None:
        _capture.append(obj)
        return
    line = encode(obj)
    with _out_lock:
        try:
            (_out or sys.stdout).write(line)
            (_out or sys.stdout).flush()
        except (BrokenPipeError, OSError, ValueError):  # the mod is gone: nobody reads us any more
            _on_broken("stdout closed")


def log(text: str) -> None:
    file_log("INFO", text)
    emit(type="log", text=text)


def warn(text: str) -> None:
    file_log("WARNING", text)
    emit(type="warn", text=text)


# =============================================================================================================
# The persistent log: log(), warn() and everything that reaches stderr (library prints, tracebacks) are also
# appended to one file, one record per line: ISO time, level, sidecar PID, text. Best effort: it never raises.
# =============================================================================================================
LOG_MAX_BYTES = 5 * 1024 * 1024  # one previous file (sidecar.log.1) is kept
_file_logger: logging.Logger | None = None
_file_broken = False
_tee: dict[str, Any] = {}
logging.addLevelName(25, "STDERR")  # library and C-level output that reached stderr
_LEVELS = {"INFO": logging.INFO, "WARNING": logging.WARNING, "STDERR": 25}


def default_log_path() -> Path:
    """Next to the model cache (speech.CACHE): LIVE_VIBE_CACHE, else $XDG_CACHE_HOME or ~/.cache, then duplex_voice."""
    base = os.environ.get("LIVE_VIBE_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "duplex_voice"
    return Path(base) / "sidecar.log"


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        when = datetime.datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")
        return f"{when} {record.levelname} pid={os.getpid()} {record.getMessage()}"


class _Handler(logging.handlers.RotatingFileHandler):
    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 - the logging API's name
        _disable_file_log("a write failed")


def _disable_file_log(why: str) -> None:
    global _file_broken
    if _file_broken:
        return
    _file_broken = True
    warn(f"sidecar log file disabled ({why}); the log and warn lines still reach the mod.")


def file_log(level: str, text: str) -> None:
    """Append text to the log file, one record per line. Silent when no file is open or it failed."""
    lg = _file_logger
    if lg is None or _file_broken:
        return
    try:
        for line in str(text).splitlines() or [""]:
            lg.log(_LEVELS.get(level, logging.INFO), line)
    except Exception:  # noqa: BLE001 - logging must never break the sidecar
        pass


def start_file_log(path: str, header: str) -> Path | None:
    """Open the log file (empty path: default_log_path()), write the session header and tee stderr into it.
    Returns the path, or None after one warn() when the file cannot be used."""
    global _file_logger
    target = Path(path).expanduser() if path else default_log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = _Handler(target, maxBytes=LOG_MAX_BYTES, backupCount=1, encoding="utf-8")
        handler.setFormatter(_Formatter())
        lg = logging.Logger("live-vibe-sidecar", logging.DEBUG)  # not in the root hierarchy: no other handler sees it
        lg.addHandler(handler)
        lg.propagate = False
        _file_logger = lg
        file_log("INFO", header)
        if _file_broken:
            return None
    except Exception as e:  # noqa: BLE001 - an unwritable path must not stop the voice
        _file_logger = None
        warn(f"sidecar log file {target} is not writable ({type(e).__name__}: {str(e)[:120]}); not logging to a file.")
        return None
    _tee_stderr()
    return target


def _tee_stderr() -> None:
    """Route fd 1 and 2 through a pipe that a thread copies to the original stderr and to the log file, so a
    library's output reaches the mod exactly as before and the file too. Any failure leaves stderr untouched."""
    try:
        sys.stderr.flush()
        r, w = os.pipe()
        orig = os.dup(2)
    except OSError:
        return
    os.dup2(w, 2)
    os.dup2(w, 1)
    _tee.update(orig=orig, w=w)

    def pump() -> None:
        pending = b""
        while True:
            try:
                chunk = os.read(r, 65536)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(orig, chunk)
            except OSError:
                pass  # the mod stopped reading stderr; the file still gets the line
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for ln in lines:
                file_log("STDERR", ln.decode("utf-8", "replace").rstrip("\r"))
        if pending:
            file_log("STDERR", pending.decode("utf-8", "replace"))
        os.close(r)

    t = threading.Thread(target=pump, name="stderr-tee", daemon=True)
    t.start()
    _tee["thread"] = t


def stop_file_log() -> None:
    """Flush the tee before the process ends (the entry point exits with os._exit, which would drop what sits in the pipe)."""
    t = _tee.pop("thread", None)
    if t is None:
        return
    try:
        sys.stderr.flush()
        os.dup2(_tee["orig"], 2)
        os.dup2(_tee["orig"], 1)
        os.close(_tee["w"])  # the last write end: the pump sees EOF
        t.join(2.0)
    except Exception:  # noqa: BLE001
        pass


# -- HTTP control server ----------------------------------------------------------------------------
# A route takes the body and says whether it was accepted: False answers 503 (a full queue).
Route = Callable[[str], bool]


def serve(routes: dict[str, Route], token: str) -> ThreadingHTTPServer:
    """POST-only control server on 127.0.0.1, an ephemeral port. Every request must carry the per-launch token in
    X-Live-Token: without it any local process (or a web page through a DNS rebind) could speak through the
    sidecar or, via /event, put words in front of Claude."""
    want = token.encode()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # a client that stalls mid-request releases its thread

        def log_message(self, *_: Any) -> None:
            pass

        def _answer(self, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            got = (self.headers.get(TOKEN_HEADER) or "").encode("utf-8", "replace")
            if not hmac.compare_digest(got, want):
                return self._answer(403)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._answer(400)
            if n < 0 or n > MAX_BODY:
                self.close_connection = True
                return self._answer(413)
            body = self.rfile.read(n).decode("utf-8", "replace")
            route = routes.get(self.path)
            if route is None:
                return self._answer(404)
            try:
                ok = route(body)
            except Exception as e:  # noqa: BLE001 - a bad request must not take the server down
                log(f"http: {self.path} failed: {type(e).__name__}: {e}")
                return self._answer(500)
            self._answer(204 if ok else 503)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    return srv


# -- lifecycle ----------------------------------------------------------------------------------------
class Lifecycle:
    """One way out. /quit, SIGTERM, SIGINT, a closed stdout and a dead parent all call request_quit(); the
    callbacks registered with on_quit() stop the session, and a watchdog ends the process QUIT_GRACE_S later
    if cleanup hangs. stdin is not watched: the mod's $.process.spawn closes it at start, so EOF there says
    nothing about the parent."""

    def __init__(self) -> None:
        self.quit = threading.Event()
        self.reason = ""
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def request_quit(self, reason: str) -> None:
        with self._lock:
            if self.quit.is_set():
                return
            self.reason = reason
            self.quit.set()
            callbacks = list(self._callbacks)
        threading.Thread(target=self._deadline, name="quit-deadline", daemon=True).start()
        for fn in callbacks:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def on_quit(self, fn: Callable[[], None]) -> None:
        with self._lock:
            if not self.quit.is_set():
                self._callbacks.append(fn)
                return
        fn()

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda n, _f: self.request_quit(signal.Signals(n).name))
        on_broken_stdout(self.request_quit)
        threading.Thread(target=self._watch_parent, name="parent-watch", daemon=True).start()

    def _deadline(self) -> None:
        time.sleep(QUIT_GRACE_S)
        try:
            sys.stderr.write(f"live sidecar: cleanup exceeded {QUIT_GRACE_S:.0f}s after {self.reason}; exiting\n")
            sys.stderr.flush()
        finally:
            os._exit(0)

    def _watch_parent(self) -> None:
        """The parent is `uv run`, itself a child of the mod: if either is killed outright, we are orphaned."""
        alive = _parent_probe()
        while not self.quit.wait(1.0):
            if not alive():
                self.request_quit("parent exited")
                return


def _parent_probe() -> Callable[[], bool]:
    ppid = os.getppid()
    if sys.platform != "win32":
        return lambda: os.getppid() == ppid  # an orphan is re-parented (to 1 or a subreaper)
    import ctypes  # Windows keeps the old ppid; ask whether that process has exited

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x00100000, False, ppid)  # SYNCHRONIZE
    if not handle:
        return lambda: True
    return lambda: kernel32.WaitForSingleObject(handle, 0) != 0  # 0: signaled, the process has exited
