"""The asyncio side of both modes: Duplex routes the listener's turn events and the HTTP routes onto the event loop;
LiveSession is /live, where the mod owns the conversation and only asks us to listen and speak."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from .audio import BACKCHANNEL_WINDOW_S, Voice, aiter_list, barge_log, is_backchannel, split_all
from .echo import words_of
from .protocol import Route, emit, log

ROUTE_WAIT_S = 2.0  # an HTTP handler waits this long for the event loop to take its request


@dataclass
class Pause:
    """A barge-in that paused the voice and waits for the user's words to say whether it is the cut."""

    at: float  # loop time of the pause
    gen: int
    final: bool = False  # BACKCHANNEL_WINDOW_S ran out first: the pause became the cut


class Duplex:
    """Shared by both modes: one inbox the listener thread posts to, the state line, a background task, and the
    two-step barge-in: speech_start pauses the voice (barge_in), and the utterance that follows resumes it when it is
    only a backchannel (or nothing: echo, noise) and ends within BACKCHANNEL_WINDOW_S; anything else, or the window
    running out, makes the pause the cut (barge_verdict, cut)."""

    def __init__(self, voice: Voice, hearing: Callable[[], bool]):
        self.voice = voice
        self.hearing = hearing  # the listener is inside a user turn
        self.loop = asyncio.get_running_loop()
        self.inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)
        self.stopped = asyncio.Event()
        self.state = ""
        self.user_talking = False
        self.pause: Pause | None = None
        self._pauses = 0
        self._spawned: set[asyncio.Task] = set()

    def spawn(self, coro) -> None:
        """Fire and forget, holding a reference so the task is not collected mid-run."""
        task = asyncio.ensure_future(coro)
        self._spawned.add(task)
        task.add_done_callback(self._spawned.discard)

    def set_state(self, s: str) -> None:
        if s != self.state:
            self.state = s
            emit(type="state", state=s)

    def settle(self) -> None:
        """The state that holds once nothing is in flight."""
        if self.user_talking:
            self.set_state("user_speaking")
        else:
            self.set_state("speaking" if self.voice.speaking.is_set() else "listening")

    # -- from other threads --------------------------------------------------------------------------------
    def post(self, kind: str, payload: Any) -> None:
        """The listener thread's way in."""
        self.loop.call_soon_threadsafe(self._put, (kind, payload))

    def _put(self, item: tuple[str, Any]) -> None:
        try:
            self.inbox.put_nowait(item)
        except asyncio.QueueFull:
            log(f"session: inbox full, dropped {item[0]}")

    def on_loop(self, fn: Callable[[str], bool]) -> Route:
        """An HTTP route that runs `fn` on the event loop and answers with its result."""
        async def call(body: str) -> bool:
            return fn(body)

        return lambda body: asyncio.run_coroutine_threadsafe(call(body), self.loop).result(ROUTE_WAIT_S)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.stopped.set)

    # -- on the loop -----------------------------------------------------------------------------------------
    async def run(self) -> None:
        background = asyncio.create_task(self.background())
        stopped = asyncio.create_task(self.stopped.wait())
        try:
            while not self.stopped.is_set():
                get = asyncio.create_task(self.inbox.get())
                await asyncio.wait({get, stopped}, return_when=asyncio.FIRST_COMPLETED)
                if not get.done():
                    get.cancel()
                    break
                try:
                    await self.handle(*get.result())
                except Exception as e:  # noqa: BLE001 - one bad event must not end the session
                    log(f"session: {type(e).__name__}: {e}")
        finally:
            await self.shutdown()
            for t in (background, stopped):
                t.cancel()
            await asyncio.wait({background, stopped}, timeout=2)

    async def background(self) -> None:
        raise NotImplementedError

    async def handle(self, kind: str, payload: Any) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Cut whatever is speaking; the caller closes devices and the server."""

    async def cut(self) -> None:
        """Make a held pause the cut: stop speaking for good."""
        raise NotImplementedError

    # -- barge-in ------------------------------------------------------------------------------------------
    def barge_in(self) -> bool:
        """speech_start over the voice: pause it and start the backchannel window. False when nothing is playing
        (the caller cuts as before)."""
        if self.pause is not None and not self.pause.final and self.voice.paused:
            return True
        if not self.voice.pause():
            return False
        self._pauses += 1
        gen = self._pauses
        self.pause = Pause(self.loop.time(), gen)

        async def due() -> None:
            await asyncio.sleep(BACKCHANNEL_WINDOW_S)
            self._put(("barge_due", gen))

        self.spawn(due())
        return True

    async def barge_due(self, gen: int) -> None:
        """The window ran out with the user still talking: the pause is the cut."""
        p = self.pause
        if p is not None and p.gen == gen and not p.final:
            p.final = True
            await self.cut()

    async def barge_verdict(self, text: str | None, discarded: bool = False) -> bool:
        """The paused barge-in's utterance decides it: True when the voice played on (the words were only a
        backchannel, or were dropped as echo or noise, and came within the window), False when the pause is the cut
        (made so here, or already). False with no pause held. Writes the barge-in's sidecar-log line."""
        p, self.pause = self.pause, None
        if p is None:
            return False
        text = (text or "").strip()
        n = len(words_of(text))
        at = self.voice.paused_ms
        within = not p.final and self.loop.time() - p.at <= BACKCHANNEL_WINDOW_S
        backchannel = bool(text) and is_backchannel(text)
        if within and (discarded or backchannel) and self.voice.resume():
            barge_log(n, at, True, "backchannel" if backchannel else "echo-ignored")
            return True
        if not p.final:
            await self.cut()
        barge_log(n, at, False, "echo-ignored" if discarded else "speech")
        return False

    def backchannel(self, text: str) -> None:
        """A backchannel the voice played on through: a note for Claude, never a prompt."""
        log(f"barge-in: {text!r} is a backchannel; the voice played on")
        emit(type="note", said=text, backchannel=True)


async def cancel_and_wait(task: asyncio.Task | None, timeout: float = 2.0) -> None:
    if task is not None and not task.done():
        task.cancel()
        await asyncio.wait({task}, timeout=timeout)


class LiveSession(Duplex):
    """/live: utterances go out to the mod, which sends Claude's answers back through POST /speak."""

    def __init__(self, voice: Voice, hearing: Callable[[], bool]):
        super().__init__(voice, hearing)
        self.speak_q: asyncio.Queue[str] = asyncio.Queue(maxsize=16)
        self.job: asyncio.Task | None = None

    def routes(self) -> dict[str, Route]:
        def speak(body: str) -> bool:
            try:
                self.speak_q.put_nowait(body)
                return True
            except asyncio.QueueFull:
                return False

        def stop(_: str) -> bool:
            self.spawn(self.stop_speaking())
            return True

        return {"/speak": self.on_loop(speak), "/stop": self.on_loop(stop)}

    async def handle(self, kind: str, payload: Any) -> None:
        if kind == "speech_start":
            self.user_talking = True
            if ((self.job and not self.job.done()) or not self.speak_q.empty()) and not self.barge_in():
                await self.stop_speaking()
            self.set_state("user_speaking")
        elif kind == "barge_due":
            await self.barge_due(payload)
        elif kind == "transcribing":
            self.set_state("transcribing")
        elif kind == "utterance":
            self.user_talking = False
            if await self.barge_verdict(payload):
                self.backchannel(payload)
            else:
                emit(type="utterance", text=payload)
            self.settle()
        elif kind == "discard":
            self.user_talking = False
            await self.barge_verdict(payload, discarded=True)
            self.settle()

    async def cut(self) -> None:
        await self.stop_speaking()

    async def stop_speaking(self) -> None:
        while not self.speak_q.empty():
            self.speak_q.get_nowait()
        emit(type="barge_in")
        await cancel_and_wait(self.job)

    async def background(self) -> None:
        """One /speak at a time, in order."""
        while True:
            text = await self.speak_q.get()
            self.job = asyncio.create_task(self.say(text))
            await asyncio.wait({self.job})

    async def say(self, text: str) -> None:
        heard: list[str] = []
        cut = True  # unless speak() returns on its own
        self.set_state("speaking")
        try:
            cut = await self.voice.speak(aiter_list(split_all(text)), heard)
        finally:
            emit(type="spoken", text=" ".join(heard), cut=cut)
            self.settle()

    async def shutdown(self) -> None:
        await cancel_and_wait(self.job)
