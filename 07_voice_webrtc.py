"""Step 7 - Skyy voice loop WITH echo cancellation (browser WebRTC).

Same pipeline as step 6, but audio goes through Chrome/Edge instead of PyAudio.
The browser applies WebRTC echo cancellation, noise suppression and auto gain
to the microphone, so Skyy can talk through the laptop speakers without
hearing itself. The page is also where the Day 7 kiosk screen can live.

PowerShell:
  $env:SKYY_JETSON = "192.168.1.50"
  python 07_voice_webrtc.py
  # then open http://localhost:7860 in Chrome or Edge, click Connect, allow the mic

Optional:
  $env:SKYY_HALF_DUPLEX = "1"   # also ignore the mic while Skyy speaks (no barge-in)
  $env:SKYY_NO_WAKE = "1"       # answer everything (default: only after "Sky ...")
  $env:SKYY_WAKE_TIMEOUT = "30" # seconds awake after the last activity (default 20)
  $env:SKYY_STT_DEVICE = "CPU"  # GPU by default; NPU on the Aura
  $env:SKYY_VOICE_ES = "es_MX-ald-medium"   # change the Spanish (or SKYY_VOICE_EN / SKYY_VOICE_FR) Piper voice
"""
import os
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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from bilingual_tts import MultilingualPiperTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from ov_whisper_stt import OpenVINOWhisperSTTService, load_whisper
from wake import DEFAULT_WAKE_WORDS, wake_word_strategies

JETSON = os.environ.get("SKYY_JETSON", "192.168.1.50")
PORT = int(os.environ.get("SKYY_LLM_PORT", "8080"))
WHISPER = os.environ.get("SKYY_WHISPER", "models/whisper-large-v3-turbo-int8-ov")
STT_DEVICE = os.environ.get("SKYY_STT_DEVICE", "GPU")
VOICE_EN = os.environ.get("SKYY_VOICE_EN", "en_US-lessac-medium")
VOICE_ES = os.environ.get("SKYY_VOICE_ES", "es_MX-claude-high")
VOICE_FR = os.environ.get("SKYY_VOICE_FR", "fr_FR-siwis-medium")

HALF_DUPLEX = os.environ.get("SKYY_HALF_DUPLEX") == "1"
WAKE = os.environ.get("SKYY_NO_WAKE") != "1"
WAKE_WORDS = os.environ.get("SKYY_WAKE_WORDS", ",".join(DEFAULT_WAKE_WORDS)).split(",")
WAKE_TIMEOUT = float(os.environ.get("SKYY_WAKE_TIMEOUT", "20"))
# Longer end-of-speech wait with the wake word, so "Sky, ... question" stays one turn
STOP_SECS = float(os.environ.get("SKYY_STOP_SECS", "0.6" if WAKE else "0.2"))

SYSTEM = ("You are Skyy, a friendly concierge for a college. Answer in the same language "
          "the visitor speaks. Keep every answer to one or two short sentences, because it "
          "will be spoken aloud. No lists, no markdown, no emojis. Only ever reply in English, "
          "Spanish or French. If the message is unclear or in another language, politely ask the "
          "visitor to repeat their question. Visitors may call you 'Sky'. If the message is only your name, greet them briefly "
          "and ask how you can help.")

# Load Whisper once at startup, not on every browser connection
WHISPER_PIPE = load_whisper(WHISPER, STT_DEVICE)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)})

    stt = OpenVINOWhisperSTTService(pipe=WHISPER_PIPE, hint="Sky" if WAKE else None)
    llm = OpenAILLMService(
        base_url=f"http://{JETSON}:{PORT}/v1", api_key="none",
        default_headers={"Connection": "close"},  # avoid stale keep-alive resets
        settings=OpenAILLMService.Settings(model="gemma", temperature=0.3, max_tokens=150))
    tts = MultilingualPiperTTSService(voices={"en": VOICE_EN, "es": VOICE_ES, "fr": VOICE_FR},
                                   download_dir=Path("models/piper"))

    context = LLMContext([{"role": "system", "content": SYSTEM}])
    agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
        user_mute_strategies=[AlwaysUserMuteStrategy()] if HALF_DUPLEX else [],
        user_turn_strategies=wake_word_strategies(
            [w.strip() for w in WAKE_WORDS if w.strip()], WAKE_TIMEOUT) if WAKE else None))

    pipeline = Pipeline([
        transport.input(), stt, agg.user(), llm, tts, transport.output(), agg.assistant()])

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_measured")
    async def on_latency(observer, seconds):
        logger.success(f"user -> bot latency: {seconds:.2f} s")

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True),
                        observers=[latency])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await task.cancel()

    logger.info(f"Connected. Half-duplex: {HALF_DUPLEX}. Wake word: {'Sky' if WAKE else 'off'}. Speak through the browser.")
    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
