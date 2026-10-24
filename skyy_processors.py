"""Two small Pipecat processors for the Skyy voice loops.

WakeWordJoiner  (between STT and the user aggregator)
    Fixes "Sky ... <pause> ... where is the library?" being split into two turns.
    When a transcript is ONLY the wake word ("Sky.", "Hey Sky!"), it is held back
    for a moment instead of going straight to the LLM:
      - if the visitor keeps talking, the next transcript is glued on and the LLM
        gets one message: "Sky. Where is the library?"  -> one answer, no greeting
      - if nobody speaks within `hold_secs`, "Sky" is released alone and Skyy greets.
    The cost is that a lone "Sky" is greeted ~1 s later.

HistoryCap  (between the user aggregator and the LLM)
    Keeps the system prompt plus at most `max_messages` messages, so old (and
    misheard) exchanges don't pile up, slow the LLM down or confuse it. It cuts in
    batches: once over the limit, history drops to the last `keep_exchanges`
    exchanges. Between cuts each request only appends to the previous one, so
    llama-server reuses its cached prompt and only processes the new answer and
    question. Dropping one exchange every turn changed the start of the history
    each time and made it reprocess the whole history (~0.3 s per turn).
"""
import asyncio
import re
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame, LLMContextFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Words that may come with the wake word without being a question ("Hey Sky", "Oye Sky")
FILLERS = {"hey", "hi", "hello", "ok", "okay", "oh", "um", "uh", "oye", "hola", "dis", "salut", "bonjour", "eh"}


class WakeWordJoiner(FrameProcessor):
    def __init__(self, wake_words, hold_secs: float = 1.2, max_hold_secs: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self._wake = {w.strip().lower() for w in wake_words if w.strip()}
        self._hold_secs = hold_secs          # wait this long for the visitor to start talking again
        self._max_hold_secs = max_hold_secs  # never hold "Sky" longer than this
        self._held: TranscriptionFrame | None = None
        self._held_at = 0.0
        self._speaking = False
        self._spoke_after_hold = False
        self._task: asyncio.Task | None = None

    def _only_wake_word(self, text: str) -> bool:
        words = re.findall(r"[a-zà-ÿ']+", text.lower())
        rest = [w for w in words if w not in FILLERS]
        return bool(rest) and all(w in self._wake for w in rest)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # VAD frames are broadcast by the aggregator in both directions; we only watch them
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            if self._held:
                self._spoke_after_hold = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._held:                                   # glue "Sky" + the question
                held, self._held = self._held, None
                await self._cancel_timer()
                frame.text = f"{held.text.strip()} {frame.text.strip()}"
                logger.info(f"Joined wake word with the question: {frame.text}")
            elif self._only_wake_word(frame.text):           # hold a lone "Sky"
                self._held, self._held_at = frame, time.monotonic()
                self._spoke_after_hold = self._speaking      # already talking again?
                self._task = self.create_task(self._release_later())
                return

        await self.push_frame(frame, direction)

    async def _release_later(self):
        while self._held:
            await asyncio.sleep(0.1)
            waited = time.monotonic() - self._held_at
            if waited > self._max_hold_secs or (waited > self._hold_secs and not self._spoke_after_hold):
                held, self._held = self._held, None
                await self.push_frame(held)                  # nobody kept talking: greet
                return
        # (if the visitor spoke, process_frame merges and clears _held)

    async def _cancel_timer(self):
        if self._task:
            await self.cancel_task(self._task)
            self._task = None

    async def cleanup(self):
        await self._cancel_timer()
        await super().cleanup()


class HistoryCap(FrameProcessor):
    def __init__(self, max_messages: int = 6, keep_exchanges: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._max = max_messages              # 6 = up to 2 earlier exchanges + the new question
        self._keep = 2 * keep_exchanges + 1   # after a cut: the last exchange(s) + the new question

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and not frame.speculation:
            msgs = frame.context.messages
            role = lambda m: m.get("role") if isinstance(m, dict) else None
            system = [m for m in msgs if role(m) == "system"]
            rest = [m for m in msgs if role(m) != "system"]
            if len(rest) > self._max:
                tail = rest[-min(self._keep, self._max):]
                while tail and role(tail[0]) != "user":   # history must start with the visitor
                    tail = tail[1:]
                frame.context.set_messages(system + tail)
        await self.push_frame(frame, direction)
