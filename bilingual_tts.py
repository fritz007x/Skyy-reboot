"""Piper TTS that picks the English, Spanish or French voice for each sentence.

Piper voices are one language each, so a single PiperTTSService reads Spanish or
French answers with the English voice. This service loads one voice per language
and, for every sentence Pipecat sends it, detects the language of that sentence's
text and speaks it with the matching voice. It looks at what Skyy is about to SAY
(the LLM's answer), so it works even when Whisper doesn't report a language.

detect_language() is also used by ov_whisper_stt.py to label transcripts.

To start speaking sooner, the first chunks of each answer are cut at a comma instead of
waiting for the end of the sentence (EarlyClauseAggregator), and Piper's leading silence
is trimmed from every chunk.

MultilingualXTTSService speaks all three languages in ONE cloned voice instead (XTTS-v2 on
the Jetson, jetson/tts_server.py). It keeps the Piper voices loaded and falls back to them
when the Jetson TTS server can't be reached, so the kiosk is never mute.

Both read URLs as their domain ("mdc.edu", not the whole link) and drop markdown symbols.
"""
import asyncio
import math
import re
import time
from pathlib import Path

import aiohttp
import numpy as np
from loguru import logger
from piper import PiperVoice
from piper.download_voices import download_voice

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.piper.tts import PiperTTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

# Common words that are distinctive for each language. Words shared by two of them
# ("la", "de", "en", "que", "a" ...) are left out on purpose: they don't help decide.
WORDS = {
    "en": set("""the an and or of to in is are was were be it this that these those for with
        on at by from as but not you your we our they their he she his her my can could would
        will what where when how which who hello hi thanks thank open until library building
        floor class classes help room today there here yes sure certainly course okay
        no""".split()),   # "no" is also Spanish: "No." alone is a tie, decided by the default
    "es": set("""el los las un una unos unas del al y o es está están son para por con sin
        como pero más muy sí no hay su sus mi tu yo usted ustedes nosotros este esta estos
        estas ese esa eso aquí allí dónde donde cuándo cuando qué cómo cuál puede puedo tiene
        tengo hola gracias buenos buenas días tardes noches abierto abierta hasta desde
        biblioteca edificio piso clase clases ayuda ayudar horario claro supuesto
        entendido perfecto""".split()),
    "fr": set("""le les des du au aux et est sont un une je tu il elle nous vous ils elles
        pour avec sans sur dans par mais pas ne plus très oui non ou ce cette ces c'est qui où
        quand comment quel quelle peut peux avez avons bonjour bonsoir merci salut ouvert
        ouverte jusqu bibliothèque bâtiment étage cours aide aider horaires aujourd'hui
        d'accord certainement entendu parfait""".split()),
}
CHARS = {
    "es": re.compile(r"[¿¡ñáíóú]", re.I),
    "fr": re.compile(r"[èêàçœùûîôëïâ]|\b[cdjlnqst]'", re.I),
}
LANGS = ("en", "es", "fr")


def detect_language(text: str, default: str = "en") -> str:
    """Return 'en', 'es' or 'fr' for a sentence; falls back to `default` when unsure."""
    lower = text.lower()
    words = re.findall(r"[a-zà-ÿœ']+", lower)
    score = {lang: sum(w in WORDS[lang] for w in words) for lang in LANGS}
    for lang, pattern in CHARS.items():
        score[lang] += 2 * len(pattern.findall(lower))
    best = max(score.values())
    winners = [lang for lang, s in score.items() if s == best]
    if best == 0 or len(winners) > 1:
        return default if default in winners or best == 0 else winners[0]
    return winners[0]


# A link is read as its domain: "https://www.mdc.edu/admissions/apply." -> "mdc.edu."
# The path may not end in punctuation, so the sentence's own period is kept.
URL = re.compile(r"\b(?:https?://)?(?:www\.)?((?:[a-z0-9-]+\.)+(?:edu|com|org|gov|net|io))"
                 r"(?:/[\w\-./?=&%#~+]*[\w/=&%#~+-])?", re.I)
MARKDOWN = re.compile(r"[*`#]+")


def clean_for_speech(text: str) -> str:
    """What the TTS should say: links as their domain, no markdown symbols."""
    text = URL.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", MARKDOWN.sub("", text)).strip()


def split_long(text: str, limit: int) -> list[str]:
    """Cut text longer than `limit` characters at a comma or semicolon (else a space)."""
    parts = []
    while len(text) > limit:
        cut = max(text.rfind(", ", 0, limit), text.rfind("; ", 0, limit))
        if cut < limit // 3:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut + 1].strip())
        text = text[cut + 1:].strip()
    parts.append(text)
    return [p for p in parts if p]


class EarlyClauseAggregator(SimpleTextAggregator):
    """Pipecat's sentence aggregator, but the first chunks of an answer may end at a comma.

    Piper only starts once it gets a whole chunk, so a long first sentence ("Para ser
    considerado estudiante de tiempo completo, debe ...") kept Skyy silent for up to
    1.3 s while the LLM wrote it. The first EARLY_CHUNKS chunks of each answer are also
    cut at ", " or "; " once they hold MIN_WORDS words and MIN_CHARS characters; after
    that Piper is already talking and later sentences are ready before they're needed.
    "1,000" (no space) and "Transcript: Request Official" (colon) are never cut, nor is a
    list of letter grades ("un W, un U, un D ou un F"): no cut after a 1-2 letter word.

    A one-word opener is always its own chunk. Pipecat's sentence splitter takes "No." for
    the abbreviation of "number" unless an English word follows, so "No. La función ..."
    used to wait for the whole sentence.

    opener_alone=False (XTTS): the one-word opener is held and sent together with the next
    chunk ("Sí. La biblioteca abre a las ocho,"), because XTTS garbles one-word inputs.
    """

    EARLY_CHUNKS = 2   # 2, not 1: a short opener ("Sure.") only plays ~0.4 s
    MIN_WORDS = 4      # shorter fragments sound choppy and are hard to tell apart by language
    MIN_CHARS = 20
    OPENER = re.compile(r"^(?!(?:dr|mr|mrs|ms|sr|sra|srta|st|mme|mlle|prof)\.)[^\W\d_]+[.!]$", re.I)

    def __init__(self, opener_alone: bool = True, **kwargs):
        self._held = ""   # before super().__init__, which may call reset()
        super().__init__(**kwargs)
        self._chunks = 0
        self.opener_alone = opener_alone

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        result = await self._next_chunk(char)
        if result is None:
            return None
        if (not self.opener_alone and not self._held and self._chunks == 0
                and self.OPENER.match(result.text)):
            self._held = result.text
            return None
        if self._held:
            result = Aggregation(text=f"{self._held} {result.text}", type=result.type)
            self._held = ""
        self._chunks += 1
        return result

    async def flush(self) -> Aggregation | None:
        held, self._held = self._held, ""   # super().flush() resets, which clears _held
        result = await super().flush()
        if held:
            return Aggregation(text=f"{held} {result.text}" if result else held,
                               type=AggregationType.SENTENCE)
        return result

    async def _next_chunk(self, char: str) -> Aggregation | None:
        result = await super()._check_sentence_with_lookahead(char)
        if result is None and self._chunks < self.EARLY_CHUNKS and char == " ":
            clause = self._text.strip()
            if self._chunks == 0 and self.OPENER.match(clause):
                cut = True
            elif self._text[-2:-1] in (",", ";"):
                cut = (len(clause) >= self.MIN_CHARS and len(clause.split()) >= self.MIN_WORDS
                       and len(clause.split()[-1]) > 3)   # last word + comma: "W," is 2 chars
            else:
                cut = False
            if cut:
                await SimpleTextAggregator.reset(self)   # empty buffer, lookahead back to idle
                result = Aggregation(text=clause, type=AggregationType.SENTENCE)
        return result

    async def handle_interruption(self):
        await super().handle_interruption()
        self._chunks = 0
        self._held = ""

    async def reset(self):   # also called by flush() at the end of each answer
        await super().reset()
        self._chunks = 0
        self._held = ""


SILENCE = 300        # int16 amplitude treated as silence when trimming
KEEP_SILENCE = 0.03  # seconds of silence left before the first sound


def trim_leading_silence(audio: bytes, sample_rate: int) -> tuple[bytes, bool]:
    """Drop leading near-silent samples, keeping KEEP_SILENCE before the first sound.
    Returns (audio, found_sound); an all-silent chunk comes back empty."""
    samples = np.frombuffer(audio, dtype=np.int16)
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > SILENCE)
    if not len(loud):
        return b"", False
    start = max(0, int(loud[0]) - int(KEEP_SILENCE * sample_rate))
    return samples[start:].tobytes(), True


class MultilingualPiperTTSService(PiperTTSService):
    def __init__(self, *, voices: dict, download_dir: Path = Path("models/piper"),
                 user_language=None, **kwargs):
        """voices: {"en": "en_US-lessac-medium", "es": "es_MX-claude-high", "fr": "fr_FR-siwis-medium"}
        user_language: returns the language of the visitor's last transcript (e.g.
        lambda: stt.last_language). An answer whose first chunk doesn't tell the language
        ("No.") is spoken in it, not in the language of the previous answer."""
        self._user_language = user_language or (lambda: None)
        self._context_id = None
        super().__init__(download_dir=download_dir,
                         settings=PiperTTSService.Settings(voice=voices["en"]), **kwargs)
        self._voices = {}
        for lang, name in voices.items():
            path = download_dir / f"{name}.onnx"
            if not path.exists():
                download_voice(name, download_dir)
            self._voices[lang] = PiperVoice.load(path)
            logger.info(f"Piper voice for '{lang}': {name}")
        self._last_lang = "en"
        # TTSService has no parameter for the aggregator, so swap the one it built.
        # Fail at startup if a Pipecat upgrade changes this, rather than silently
        # going back to whole sentences.
        assert isinstance(self._text_aggregator, SimpleTextAggregator), \
            "Pipecat changed TTSService._text_aggregator; update EarlyClauseAggregator"
        self._text_aggregator = EarlyClauseAggregator(
            aggregation_type=self._text_aggregator.aggregation_type,
            language=self._text_aggregator.language)

    def _pick_language(self, text: str, context_id: str) -> str:
        if context_id != self._context_id:   # first chunk of a new answer
            self._context_id = context_id
            if self._user_language() in self._voices:
                self._last_lang = self._user_language()
        lang = detect_language(text, default=self._last_lang)
        if lang not in self._voices:
            lang = "en"
        self._last_lang = lang
        return lang

    async def run_tts(self, text: str, context_id: str):
        text = clean_for_speech(text)
        if not text:
            return
        lang = self._pick_language(text, context_id)
        logger.debug(f"{self}: speaking [{lang}] {text}")
        trimming = True   # until the first non-silent audio of this chunk
        async for frame in self._synthesize(text, lang, context_id):
            if trimming and isinstance(frame, TTSAudioRawFrame):
                audio, trimming_done = trim_leading_silence(frame.audio, frame.sample_rate)
                if not audio:
                    continue   # all silence: drop the frame
                frame.audio, frame.num_frames = audio, len(audio) // (2 * frame.num_channels)
                trimming = not trimming_done
            yield frame

    async def _synthesize(self, text: str, lang: str, context_id: str):
        """Audio frames for one chunk of text, in language `lang` (Piper voice of that language)."""
        self._voice = self._voices[lang]
        async for frame in PiperTTSService.run_tts(self, text, context_id):
            yield frame


class MultilingualXTTSService(MultilingualPiperTTSService):
    """One cloned voice for English, Spanish and French: XTTS-v2 on the Jetson (tts_server.py).

    Same language choice per chunk, early clauses and silence trimming as the Piper service;
    only the synthesis goes to the server. The Piper voices stay loaded as the fallback: if
    the server can't be reached or returns an error, that chunk and the next RETRY_SECS of
    speech use Piper, then the server is tried again.
    """

    SAMPLE_RATE = 24000                          # XTTS output; resampled to the pipeline rate
    MAX_CHARS = {"en": 250, "es": 239, "fr": 273}  # XTTS's per-input limits
    RETRY_SECS = 30

    def __init__(self, *, base_url: str, voice: str, **kwargs):
        """base_url: e.g. "http://192.168.55.1:8083"; voice: a voice folder on the server
        ("jabberjaw"). Other arguments as MultilingualPiperTTSService (Piper fallback voices)."""
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._xtts_voice = voice
        self._session: aiohttp.ClientSession | None = None
        self._down_until = 0.0
        self._text_aggregator.opener_alone = False   # XTTS garbles one-word inputs ("Sí.")
        # Pipecat's stream resampler is tied to one input rate: XTTS is 24 kHz, Piper 22.05 kHz
        self._piper_resampler, self._xtts_resampler = self._resampler, create_stream_resampler()

    async def start(self, frame):
        await super().start(frame)
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=1, sock_read=10))
        await self._check_server()

    async def stop(self, frame):
        await super().stop(frame)
        await self._close()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close()

    async def _close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _check_server(self):
        try:
            async with self._session.get(self._base_url + "/health",
                                         timeout=aiohttp.ClientTimeout(total=3)) as r:
                voices = (await r.json())["voices"]
        except Exception as e:
            logger.warning(f"{self}: XTTS server {self._base_url} not reachable ({e}); "
                           f"speaking with Piper until it is")
            self._down_until = time.monotonic() + self.RETRY_SECS
            return
        if self._xtts_voice in voices:
            logger.info(f"XTTS voice '{self._xtts_voice}' at {self._base_url} for en/es/fr")
        else:
            logger.error(f"{self}: XTTS server has no voice '{self._xtts_voice}' (has {voices}); "
                         f"speaking with Piper")
            self._down_until = math.inf

    async def _piper(self, text: str, lang: str, context_id: str):
        self._resampler = self._piper_resampler
        async for frame in super()._synthesize(text, lang, context_id):
            yield frame

    async def _synthesize(self, text: str, lang: str, context_id: str):
        if self._session is None or time.monotonic() < self._down_until:
            async for frame in self._piper(text, lang, context_id):
                yield frame
            return
        parts = split_long(text, self.MAX_CHARS.get(lang, 250))
        done = 0
        self._resampler = self._xtts_resampler
        try:
            for part in parts:
                async with self._session.post(self._base_url + "/tts_stream", json={
                        "text": part, "language": lang, "voice": self._xtts_voice}) as r:
                    if r.status != 200:
                        raise RuntimeError(f"HTTP {r.status}: {await r.text()}")
                    await self.start_tts_usage_metrics(part)
                    async for frame in self._stream_audio_frames_from_iterator(
                            r.content.iter_any(), in_sample_rate=self.SAMPLE_RATE,
                            context_id=context_id):
                        await self.stop_ttfb_metrics()
                        yield frame
                done += 1
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"{self}: XTTS failed ({e}); Piper for the next {self.RETRY_SECS} s")
            self._down_until = time.monotonic() + self.RETRY_SECS
            rest = " ".join(parts[done:])
            async for frame in self._piper(rest, lang, context_id):
                yield frame
        finally:
            await self.stop_ttfb_metrics()


BilingualPiperTTSService = MultilingualPiperTTSService  # old name still works
