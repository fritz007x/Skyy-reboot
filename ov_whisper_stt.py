"""Pipecat STT service that runs Whisper through OpenVINO GenAI.

Pipecat's built-in WhisperSTTService uses faster-whisper (CPU/CUDA only). This one
runs on Intel CPU, iGPU or NPU (device="NPU" on the Aura; "GPU" and "CPU" are fallbacks).

Language handling: Whisper's automatic language detection picks from ~100 languages
and can mistake a short Spanish question for e.g. Urdu. Skyy only supports English,
Spanish and French, so if the automatic result isn't one of those (wrong language code, or
text in a non-Latin script), the audio is re-transcribed forced to each allowed
language and the more confident result is kept.
"""
import asyncio
import os
import re
import time

import numpy as np
import openvino_genai as ov_genai
from loguru import logger

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from bilingual_tts import detect_language

# Letters, digits, punctuation and accents used by English/Spanish/French (Latin script)
LATIN_ONLY = re.compile(r"^[\sA-Za-z0-9À-ÿœŒ.,;:!?¿¡«»’'\"()\-–—%$&/@#+*]*$")


CACHE_DIR = os.environ.get("SKYY_OV_CACHE", "ov_cache")   # compiled models, so NPU/GPU start fast


def load_whisper(model_dir: str, device: str = "GPU"):
    """Load and warm up the model once; share it between service instances.

    device: "NPU", "GPU" or "CPU". If it fails to load (e.g. NPU driver missing), falls back
    to the next device in NPU -> GPU -> CPU so the concierge still starts.
    The first NPU load compiles the model (can take a minute or more); the result is cached
    in ov_cache/, so later starts take seconds."""
    order = ["NPU", "GPU", "CPU"]
    tries = [device] + [d for d in order[order.index(device) + 1:]] if device in order else [device, "CPU"]
    last = None
    for dev in tries:
        try:
            t = time.perf_counter()
            logger.info(f"Loading OpenVINO Whisper from {model_dir} on {dev} ...")
            try:
                pipe = ov_genai.WhisperPipeline(model_dir, dev, CACHE_DIR=CACHE_DIR)
            except TypeError:   # older OpenVINO without keyword properties
                pipe = ov_genai.WhisperPipeline(model_dir, dev)
            pipe.generate(np.zeros(16000, dtype=np.float32).tolist())  # warm-up / compile
            logger.info(f"Whisper ready on {dev} in {time.perf_counter() - t:.1f} s")
            return pipe
        except Exception as e:
            last = e
            logger.warning(f"Whisper could not load on {dev}: {e}")
    raise RuntimeError(f"Whisper failed on every device: {last}")


def _lang_code(res) -> str:
    return str((getattr(res, "languages", None) or [""])[0]).strip("<|>").lower()


def _confidence(res) -> float:
    """Average log-probability per generated token (higher = more confident)."""
    try:
        n = max(1, res.perf_metrics.get_num_generated_tokens())
        return float(res.scores[0]) / n
    except Exception:
        return float("-inf")


class OpenVINOWhisperSTTService(SegmentedSTTService):
    def __init__(self, *, model_dir: str = "", device: str = "GPU", pipe=None,
                 languages=("en", "es", "fr"), hint: str | None = None, **kwargs):
        """hint: words Whisper should expect, e.g. "Sky" for the wake word and names like
        "Gordon Rule". Passed as Whisper's initial prompt, it makes "Sky" far less likely to
        come out as "Hi" and "Gordon Rule" as "corner rule"."""
        super().__init__(settings=STTSettings(model=model_dir or "openvino-whisper",
                                              language=None), **kwargs)
        self._pipe = pipe or load_whisper(model_dir, device)
        self._languages = tuple(languages)
        self._gen = {"task": "transcribe"}
        if hint:
            self._gen["initial_prompt"] = hint
        self.last_language: str | None = None

    @property
    def wants_wav_segments(self) -> bool:
        return False  # give us raw 16-bit PCM

    def can_generate_metrics(self) -> bool:
        return True

    def _generate(self, speech, **kw):
        """pipe.generate with our options. If this device/OpenVINO version rejects
        initial_prompt (possible on NPU), drop the hint once and carry on without it.

        Output is capped at ~10 tokens per second of audio: a hallucination loop
        ("δοδοδο...") then stops early instead of running to Whisper's 448-token limit,
        which took 2.8 s once. Normal speech is 3-5 tokens per second."""
        kw.setdefault("max_new_tokens", min(440, int(len(speech) / 16000 * 10) + 20))
        try:
            return self._pipe.generate(speech, **self._gen, **kw)
        except Exception as e:
            if "initial_prompt" not in self._gen:
                raise
            logger.warning(f"Whisper rejected the hint on this device ({e}); continuing without it")
            self._gen.pop("initial_prompt")
            return self._pipe.generate(speech, **self._gen, **kw)

    def _transcribe(self, speech: list[float]):
        res = self._generate(speech)
        text, code = res.texts[0].strip(), _lang_code(res)
        ok_lang = code in self._languages if code else True   # no code reported -> judge by text
        if ok_lang and LATIN_ONLY.match(text):
            if not code:  # this OpenVINO version doesn't report the language: guess from text
                code = detect_language(text)
            return text, code, False
        # Auto-detect went outside the allowed languages.
        if len(speech) < 2.0 * 16000:
            # Short clip (a word or two, often just "Sky"): one forced pass in the language the
            # visitor last used (English at first) instead of 3 passes, which take ~5 s and
            # delay whatever the visitor says next.
            lang = self.last_language if self.last_language in self._languages else "en"
            r = self._generate(speech, language=f"<|{lang}|>")
            logger.warning(f"Whisper auto-detect gave [{code or '?'}] '{text}' (short clip) -> "
                           f"re-transcribed as [{lang}] '{r.texts[0].strip()}'")
            return r.texts[0].strip(), lang, True
        # Longer clip: try each allowed language, keep the most confident result
        best = None
        for lang in self._languages:
            r = self._generate(speech, language=f"<|{lang}|>")
            cand = (_confidence(r), r.texts[0].strip(), lang)
            if best is None or cand[0] > best[0]:
                best = cand
        logger.warning(f"Whisper auto-detect gave [{code or '?'}] '{text}' -> "
                       f"re-transcribed as [{best[2]}] '{best[1]}'")
        return best[1], best[2], True

    async def run_stt(self, audio: bytes):
        await self.start_processing_metrics()
        speech = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if self.sample_rate and self.sample_rate != 16000:
            from scipy.signal import resample_poly
            g = np.gcd(self.sample_rate, 16000)
            speech = resample_poly(speech, 16000 // g, self.sample_rate // g).astype(np.float32)
        if os.environ.get("SKYY_SAVE_AUDIO"):  # debug: keep exactly what Whisper receives
            import soundfile as sf
            os.makedirs("debug_audio", exist_ok=True)
            self._last_wav = os.path.join("debug_audio", time.strftime("%H%M%S") + ".wav")
            sf.write(self._last_wav, speech, 16000)
        try:
            text, code, _ = await asyncio.to_thread(self._transcribe, speech.tolist())
        except Exception as e:
            yield ErrorFrame(f"OpenVINO Whisper failed: {e}")
            return
        await self.stop_processing_metrics()

        try:
            language = Language(code)
        except ValueError:
            language = None
        self.last_language = code
        if text:
            saved = f"  (audio: {self._last_wav})" if os.environ.get("SKYY_SAVE_AUDIO") else ""
            logger.info(f"Heard [{code}]: {text}{saved}")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
