"""Wake word "Sky" for the Skyy voice loops.

Uses Pipecat's WakePhraseUserTurnStartStrategy, which checks each Whisper
transcript for the wake word. While asleep, everything else people say is
transcribed and then dropped (nothing reaches the LLM). After "Sky ..." Skyy
stays awake until `timeout` seconds after its last answer, so follow-up questions
don't need the wake word again. Other people's speech doesn't extend that window
(see SkyyWakeStrategy), and SideTalkFilter (skyy_processors.py) drops fragments
of side conversation that arrive while Skyy is awake. Works in English, Spanish and French because it
matches the transcript, not the sound.

"Sky, where is the library?" -> wakes and answers the question in one go.
"Sky."                        -> wakes; Skyy greets and waits for the question.

Pause tip: a pause after "Sky," can end the turn early, splitting the question off.
The loops therefore use a longer end-of-speech wait (0.6 s) when the wake word is on.
"""
import asyncio
import time

from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import (
    BotSpeakingFrame, TranscriptionFrame, UserSpeakingFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import WakePhraseUserTurnStartStrategy
from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import _WakeState
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies, default_user_turn_start_strategies

# How Whisper tends to spell "Sky"/"Skyy" in English, Spanish and French transcripts.
# ("ski" is left out on purpose: too many false wakes, e.g. "ski club".)
DEFAULT_WAKE_WORDS = ["sky", "skye", "skyy", "skies", "skype", "skai", "scai", "escai"]


class SkyyWakeStrategy(WakePhraseUserTurnStartStrategy):
    """Pipecat's wake phrase strategy, but only Skyy's own speech keeps it awake.

    Pipecat restarts the awake timer on any speech, so people talking near the kiosk (or on
    a Zoom call) kept Skyy awake for good and it answered their side talk. Here the full
    `timeout` starts at the wake word and each time Skyy speaks. The visitor's speech only
    keeps Skyy awake until `grace` seconds after they stop, long enough for the transcript
    to arrive. A follow-up question within `timeout` of an answer needs no wake word.
    """

    def __init__(self, *, grace: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self._grace = grace
        self._deadline = 0.0

    def _refresh_timeout(self):
        self._deadline = time.monotonic() + self._timeout

    def _extend(self):
        self._deadline = max(self._deadline, time.monotonic() + self._grace)

    async def handle_user_turn_started(self):
        if self._state == _WakeState.AWAKE:
            self._extend()

    async def _process_awake(self, frame) -> ProcessFrameResult:
        if isinstance(frame, BotSpeakingFrame):
            self._refresh_timeout()
        elif isinstance(frame, (UserSpeakingFrame, VADUserStartedSpeakingFrame,
                                VADUserStoppedSpeakingFrame, TranscriptionFrame)):
            self._extend()
        return ProcessFrameResult.CONTINUE

    async def _timeout_task_handler(self):
        while True:
            await asyncio.sleep(0.2)
            if self._state == _WakeState.AWAKE and time.monotonic() > self._deadline:
                self._transition_to_idle()


def wake_word_strategies(words=DEFAULT_WAKE_WORDS, timeout: float = 6.0,
                         incomplete_secs: float = 1.5) -> UserTurnStrategies:
    """timeout: seconds Skyy stays awake after the wake word or after it last spoke.
    incomplete_secs: silence after which a turn Smart Turn judged INCOMPLETE ends anyway
    (Pipecat's default is 3 s, which made "How do I get a new one?" wait 3 s)."""
    wake = SkyyWakeStrategy(phrases=list(words), timeout=timeout)

    @wake.event_handler("on_wake_phrase_detected")
    async def _on_wake(strategy, phrase):
        logger.success(f"Wake word heard ('{phrase}') - listening until {timeout:.0f} s after each answer")

    @wake.event_handler("on_wake_phrase_timeout")
    async def _on_sleep(strategy):
        logger.info("No activity - going back to sleep. Say 'Sky' to wake me.")

    smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=incomplete_secs))
    return UserTurnStrategies(start=[wake, *default_user_turn_start_strategies()],
                              stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)])
