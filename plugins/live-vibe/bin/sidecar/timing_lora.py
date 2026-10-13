"""The prompt the published timing LoRA was trained on (training/README.md), as one fingerprint.

The adapter learned one front prompt, one micro-turn protocol, one window format and one set of history messages.
A change to any of them leaves it loading fine but timing text it never saw, and nothing else shows that. So
`--unit` fails while fingerprint() differs from TRAINED, and the front warns at start when it serves the named
adapter on a changed prompt. The fix is a retrain with make_windows.py on the new prompt, a new upload, and the new
TRAINED, revision and sha256 (front_server.TIMING_LORA) in one commit.
"""
from __future__ import annotations

import hashlib
import json

from .front import FRONT_PROMPT_EXPERIMENTAL, HOLD, RETELL_STATUS, event_message, notes_message
from .microturn import FORMAT, MICRO_PROTOCOL, NO_VOICE, PREFIX, SPEAK, json_turn, window_message

# fingerprint() of the code that built the published adapter's training data (live-vibe 0.7.2).
TRAINED = "f7939ba04ef964faf2f72bf4dac2eeecfe2e615974792943ea42a7a7f09a2254"


def fingerprint() -> str:
    """sha256 over every piece of text make_windows.py puts in a training row that the sidecar also sends: the
    system prompt's template, the protocol, the action prefix and schema, and one of each message shape."""
    parts = [
        FRONT_PROMPT_EXPERIMENTAL,
        MICRO_PROTOCOL,
        PREFIX,
        json.dumps(FORMAT, sort_keys=True),
        window_message("can you run the tests", 0.83, False),
        window_message("", None, True),
        NO_VOICE,
        json_turn(SPEAK, "run the tests", "On it."),
        event_message("The tests pass.", RETELL_STATUS),
        notes_message("background"),
        HOLD,
    ]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def stale() -> bool:
    return fingerprint() != TRAINED
