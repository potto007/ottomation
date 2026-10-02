"""The asyncio side of both modes: Duplex routes the listener's turn events and the HTTP routes onto the event loop;
LiveSession is /live, where the mod owns the conversation and only asks us to listen and speak."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from .audio import Voice, aiter_list, split_all
from .protocol import Route, emit, log

ROUTE_WAIT_S = 2.0  # an HTTP handler waits this long for the event loop to take its request


class Duplex:
    """Shared by both modes: one inbox the listener thread posts to, the state line, and a background task."""

    def __init__(self, voice: Voice, hearing: Callable[[], bool]):
        self.voice = voice
        self.hearing = hearing  # the listener is inside a user turn
        self.loop = asyncio.get_running_loop()
        self.inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)
        self.stopped = asyncio.Event()
        self.state = ""
        self.user_talking = False
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
            if (self.job and not self.job.done()) or not self.speak_q.empty():
                await self.stop_speaking()
            self.set_state("user_speaking")
        elif kind == "transcribing":
            self.set_state("transcribing")
        elif kind == "utterance":
            self.user_talking = False
            emit(type="utterance", text=payload)
            self.settle()
        elif kind == "discard":
            self.user_talking = False
            self.settle()

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
