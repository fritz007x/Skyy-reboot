"""Step 7 - Skyy voice loop WITH echo cancellation (browser WebRTC).

Same pipeline as step 6, but audio goes through Chrome/Edge instead of PyAudio.
The browser applies WebRTC echo cancellation, noise suppression and auto gain
to the microphone, so Skyy can talk through the laptop speakers without
hearing itself. The page is also where the Day 7 kiosk screen can live.

PowerShell:
  $env:SKYY_JETSON = "192.168.55.1"   # USB-C link
  python 07_voice_webrtc.py
  # then open http://localhost:7860 in Chrome or Edge, click Connect, allow the mic

Optional:
  $env:SKYY_HALF_DUPLEX = "1"   # also ignore the mic while Skyy speaks (no barge-in)
  $env:SKYY_NO_WAKE = "1"       # answer everything (default: only after "Sky ...")
  $env:SKYY_WAKE_TIMEOUT = "30" # seconds awake after the last activity (default 10)
  $env:SKYY_WAKE_WORDS = "sky,skye,hey sky"   # spellings that count as the wake word
  $env:SKYY_STOP_SECS = "0.4"   # silence that ends a turn (default 0.4 with wake word, 0.2 without)
  $env:SKYY_INCOMPLETE_SECS = "1.5"  # ...when Smart Turn thinks the sentence isn't finished (default 1.5)
  $env:SKYY_HISTORY = "6"       # messages of conversation the LLM sees (last 3 Q&A)
  $env:SKYY_STT_DEVICE = "NPU"  # GPU by default; NPU on the Aura (falls back to GPU, then CPU)
  $env:SKYY_STT_HINT = "MDC, Gordon Rule"   # names Whisper should expect (default: the FAQ's proper nouns)
  $env:SKYY_VOICE_ES = "es_MX-ald-medium"   # change the Spanish (or SKYY_VOICE_EN / SKYY_VOICE_FR) Piper voice

Knowledge base (MDC FAQ, see kb.py; build the Qdrant collection once with 08_kb_index.py):
  $env:SKYY_KB_MODE = "prompt"  # prompt (default): whole FAQ in the cached system prompt; rag: EmbeddingGemma + Qdrant per question
                                # off: no knowledge base
  $env:SKYY_QDRANT_KEY = "..."  # Qdrant API key (rag mode)
  $env:SKYY_KB_TOP_K = "3"      # FAQ entries added per question (rag mode)
  $env:SKYY_KB_MIN_SCORE = "0.67"  # ignore weaker matches (rag mode; 0.67 suits EmbeddingGemma 2 on this FAQ)
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

import kb
from skyy_processors import HistoryCap, WakeWordJoiner
from ov_whisper_stt import OpenVINOWhisperSTTService, load_whisper
from wake import DEFAULT_WAKE_WORDS, wake_word_strategies

JETSON = os.environ.get("SKYY_JETSON", "192.168.1.50")
PORT = int(os.environ.get("SKYY_LLM_PORT", "8080"))
WHISPER = os.environ.get("SKYY_WHISPER", "models/whisper-large-v3-turbo-int8-ov")
STT_DEVICE = os.environ.get("SKYY_STT_DEVICE", "GPU")
# Proper nouns from the FAQ that Whisper misheard ("corner rule", "Miami Day"). Only names,
# no full sentences: the hint is Whisper's initial prompt and could bias its language detection.
STT_HINT = os.environ.get("SKYY_STT_HINT",
                          "Miami Dade College, MDC, MDConnect, Gordon Rule, MAP, REVEST, FAFSA.")
VOICE_EN = os.environ.get("SKYY_VOICE_EN", "en_US-lessac-medium")
VOICE_ES = os.environ.get("SKYY_VOICE_ES", "es_MX-claude-high")
VOICE_FR = os.environ.get("SKYY_VOICE_FR", "fr_FR-siwis-medium")

HALF_DUPLEX = os.environ.get("SKYY_HALF_DUPLEX") == "1"
WAKE = os.environ.get("SKYY_NO_WAKE") != "1"
WAKE_WORDS = os.environ.get("SKYY_WAKE_WORDS", ",".join(DEFAULT_WAKE_WORDS)).split(",")
WAKE_TIMEOUT = float(os.environ.get("SKYY_WAKE_TIMEOUT", "10"))
# Longer end-of-speech wait with the wake word, so "Sky, ... question" stays one turn
HISTORY = int(os.environ.get("SKYY_HISTORY", "6"))  # messages the LLM sees (last 3 Q&A)
STOP_SECS = float(os.environ.get("SKYY_STOP_SECS", "0.4" if WAKE else "0.2"))
# Silence after which a turn Smart Turn calls INCOMPLETE ends anyway (Pipecat default: 3)
INCOMPLETE_SECS = float(os.environ.get("SKYY_INCOMPLETE_SECS", "1.5"))

from skyy_prompt import SYSTEM
SYSTEM = kb.system_prompt(SYSTEM)   # adds the FAQ rules (and in prompt mode the whole FAQ)

# Load Whisper once at startup, not on every browser connection
WHISPER_PIPE = load_whisper(WHISPER, STT_DEVICE)
# Have llama-server process and cache the system prompt now, not on the first question
kb.warm_up(JETSON, PORT, SYSTEM)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)})

    stt = OpenVINOWhisperSTTService(pipe=WHISPER_PIPE,
                                    hint=" ".join(filter(None, ["Sky." if WAKE else "", STT_HINT])) or None)
    llm = OpenAILLMService(
        base_url=f"http://{JETSON}:{PORT}/v1", api_key="none",
        default_headers={"Connection": "close"},  # avoid stale keep-alive resets
        settings=OpenAILLMService.Settings(model="gemma", temperature=0.3, max_tokens=250))
    tts = MultilingualPiperTTSService(voices={"en": VOICE_EN, "es": VOICE_ES, "fr": VOICE_FR},
                                   download_dir=Path("models/piper"),
                                   user_language=lambda: stt.last_language)

    context = LLMContext([{"role": "system", "content": SYSTEM}])
    agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
        user_mute_strategies=[AlwaysUserMuteStrategy()] if HALF_DUPLEX else [],
        user_turn_strategies=wake_word_strategies(
            [w.strip() for w in WAKE_WORDS if w.strip()], WAKE_TIMEOUT, INCOMPLETE_SECS) if WAKE else None))

    # "Sky ... <pause> ... question" -> one turn; lone "Sky" -> greeting (see skyy_processors.py)
    joiner = [WakeWordJoiner([w.strip() for w in WAKE_WORDS if w.strip()])] if WAKE else []
    # rag mode: add the closest FAQ entries to each question (see kb.py)
    rag = [kb.KnowledgeInjector(kb.KnowledgeBase(JETSON), strip_words=WAKE_WORDS)] if kb.KB_MODE == "rag" else []
    pipeline = Pipeline([
        transport.input(), stt, *joiner, agg.user(), HistoryCap(HISTORY), *rag,
        llm, tts, transport.output(), agg.assistant()])

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_measured")
    async def on_latency(observer, seconds):
        logger.success(f"user -> bot latency: {seconds:.2f} s")

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True),
                        observers=[latency])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await task.cancel()

    logger.info(f"Connected. Half-duplex: {HALF_DUPLEX}. Wake word: {'Sky' if WAKE else 'off'}. "
                f"Knowledge base: {kb.KB_MODE}. Speak through the browser.")
    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
