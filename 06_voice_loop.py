"""Step 6 - minimal Skyy voice loop with Pipecat (Day 2 "done when").

mic -> Silero VAD -> Whisper (OpenVINO) -> Gemma on the Jetson -> Piper -> speaker

  python 06_voice_loop.py --host 192.168.1.50
  python 06_voice_loop.py --host 192.168.1.50 --stt-device CPU --voice-es es_MX-ald-medium

ECHO: this local transport has no echo cancellation. By default the mic is ignored
while Skyy speaks (half-duplex), so it works on laptop speakers without Skyy hearing
and answering itself. With headphones you can pass --barge-in to allow interrupting
Skyy mid-answer. For speakers + interruptions, use step 7 (browser echo cancellation).
WAKE WORD: Skyy only answers after hearing "Sky" (e.g. "Sky, where is the library?"),
then keeps listening for --wake-timeout seconds of inactivity. --no-wake to disable.
Ctrl+C to stop.
After each answer, the log shows "user -> bot latency" (end of your speech to
first audio) plus a per-service breakdown.
"""
import argparse, asyncio
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams)
from pipecat.services.openai.llm import OpenAILLMService
from bilingual_tts import MultilingualPiperTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from skyy_processors import HistoryCap, WakeWordJoiner
from ov_whisper_stt import OpenVINOWhisperSTTService
from wake import DEFAULT_WAKE_WORDS, wake_word_strategies

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, help="Jetson IP")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--whisper", default="models/whisper-large-v3-turbo-int8-ov",
                help="folder downloaded by 03_whisper.py")
ap.add_argument("--stt-device", default="GPU", help="CPU, GPU (or NPU on the Aura)")
ap.add_argument("--voice-en", default="en_US-lessac-medium", help="Piper voice for English")
ap.add_argument("--voice-es", default="es_MX-claude-high", help="Piper voice for Spanish")
ap.add_argument("--voice-fr", default="fr_FR-siwis-medium", help="Piper voice for French")
ap.add_argument("--stop-secs", type=float, default=None,
                help="silence that ends your turn (default 0.6 with wake word, 0.2 without)")
ap.add_argument("--barge-in", action="store_true",
                help="let the user interrupt Skyy (only with headphones: no echo cancellation here)")
ap.add_argument("--no-wake", action="store_true", help="answer everything, no wake word needed")
ap.add_argument("--wake-words", default=",".join(DEFAULT_WAKE_WORDS),
                help="comma-separated spellings of the wake word as Whisper writes them")
ap.add_argument("--wake-timeout", type=float, default=20.0,
                help="seconds of silence before Skyy needs the wake word again")
ap.add_argument("--history", type=int, default=6,
                help="messages of conversation the LLM sees (6 = last 3 questions + answers)")
ap.add_argument("--in-device", type=int, help="PyAudio input index (see README)")
ap.add_argument("--out-device", type=int)
a = ap.parse_args()
if a.stop_secs is None:  # longer wait with wake word, so "Sky, ... question" stays one turn
    a.stop_secs = 0.2 if a.no_wake else 0.6

SYSTEM = ("You are Skyy, a friendly concierge for a college. Answer in the same language "
          "the visitor speaks. Keep every answer to one or two short sentences, because it "
          "will be spoken aloud. No lists, no markdown, no emojis. Only ever reply in English, "
          "Spanish or French. If the message is unclear or in another language, politely ask the "
          "visitor to repeat their question. Visitors may call you 'Sky'. If the message is only your name, greet them briefly "
          "and ask how you can help.")


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True,
        input_device_index=a.in_device, output_device_index=a.out_device))

    # With the wake word on, tell Whisper to expect "Sky" (otherwise it often hears "Hi")
    stt = OpenVINOWhisperSTTService(model_dir=a.whisper, device=a.stt_device,
                                    hint=None if a.no_wake else "Sky")

    llm = OpenAILLMService(
        base_url=f"http://{a.host}:{a.port}/v1", api_key="none",
        default_headers={"Connection": "close"},  # avoid stale keep-alive resets
        settings=OpenAILLMService.Settings(model="gemma", temperature=0.3, max_tokens=150))

    # One Piper voice per language; each sentence is spoken in the voice of its language
    tts = MultilingualPiperTTSService(voices={"en": a.voice_en, "es": a.voice_es, "fr": a.voice_fr},
                                   download_dir=Path("models/piper"))

    context = LLMContext([{"role": "system", "content": SYSTEM}])
    agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=a.stop_secs)),
        # Half-duplex: ignore the mic while Skyy is talking, so it can't hear itself
        user_mute_strategies=[] if a.barge_in else [AlwaysUserMuteStrategy()],
        # Wake word: only talk to Skyy after "Sky ..." (see wake.py)
        user_turn_strategies=None if a.no_wake else wake_word_strategies(
            [w.strip() for w in a.wake_words.split(",") if w.strip()], a.wake_timeout)))

    words = [w.strip() for w in a.wake_words.split(",") if w.strip()]
    # "Sky ... <pause> ... question" -> one turn; lone "Sky" -> greeting (see skyy_processors.py)
    joiner = [] if a.no_wake else [WakeWordJoiner(words)]
    pipeline = Pipeline([
        transport.input(), stt, *joiner, agg.user(), HistoryCap(a.history),
        llm, tts, transport.output(), agg.assistant()])

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_measured")
    async def on_latency(observer, seconds):
        logger.success(f"user -> bot latency: {seconds:.2f} s")

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True),
                        observers=[latency])
    logger.info(("Ready - speak freely." if a.no_wake else "Asleep - say 'Sky' to wake me (e.g. 'Sky, where is the library?').") + f" Barge-in: {a.barge_in} (mic muted while Skyy talks: {not a.barge_in}). Ctrl+C to stop.")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
