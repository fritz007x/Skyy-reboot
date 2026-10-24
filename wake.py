"""Wake word "Sky" for the Skyy voice loops.

Uses Pipecat's WakePhraseUserTurnStartStrategy, which checks each Whisper
transcript for the wake word. While asleep, everything else people say is
transcribed and then dropped (nothing reaches the LLM). After "Sky ..." Skyy
stays awake for `timeout` seconds of inactivity, so follow-up questions don't
need the wake word again. Works in English, Spanish and French because it
matches the transcript, not the sound.

"Sky, where is the library?" -> wakes and answers the question in one go.
"Sky."                        -> wakes; Skyy greets and waits for the question.

Pause tip: a pause after "Sky," can end the turn early, splitting the question off.
The loops therefore use a longer end-of-speech wait (0.6 s) when the wake word is on.
"""
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_start import WakePhraseUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies, default_user_turn_start_strategies

# How Whisper tends to spell "Sky"/"Skyy" in English, Spanish and French transcripts.
# ("ski" is left out on purpose: too many false wakes, e.g. "ski club".)
DEFAULT_WAKE_WORDS = ["sky", "skye", "skyy", "skies", "skype", "skai", "scai", "escai"]


def wake_word_strategies(words=DEFAULT_WAKE_WORDS, timeout: float = 20.0,
                         incomplete_secs: float = 1.5) -> UserTurnStrategies:
    """incomplete_secs: silence after which a turn Smart Turn judged INCOMPLETE ends anyway
    (Pipecat's default is 3 s, which made "How do I get a new one?" wait 3 s)."""
    wake = WakePhraseUserTurnStartStrategy(phrases=list(words), timeout=timeout)

    @wake.event_handler("on_wake_phrase_detected")
    async def _on_wake(strategy, phrase):
        logger.success(f"Wake word heard ('{phrase}') - listening for {timeout:.0f} s")

    @wake.event_handler("on_wake_phrase_timeout")
    async def _on_sleep(strategy):
        logger.info("No activity - going back to sleep. Say 'Sky' to wake me.")

    smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=incomplete_secs))
    return UserTurnStrategies(start=[wake, *default_user_turn_start_strategies()],
                              stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)])
