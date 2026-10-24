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
"""
import re
from pathlib import Path

import numpy as np
from loguru import logger
from piper import PiperVoice
from piper.download_voices import download_voice

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
    """

    EARLY_CHUNKS = 2   # 2, not 1: a short opener ("Sure.") only plays ~0.4 s
    MIN_WORDS = 4      # shorter fragments sound choppy and are hard to tell apart by language
    MIN_CHARS = 20
    OPENER = re.compile(r"^(?!(?:dr|mr|mrs|ms|sr|sra|srta|st|mme|mlle|prof)\.)[^\W\d_]+[.!]$", re.I)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunks = 0

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
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
                await super().reset()   # empty buffer, lookahead back to idle
                result = Aggregation(text=clause, type=AggregationType.SENTENCE)
        if result is not None:
            self._chunks += 1
        return result

    async def handle_interruption(self):
        await super().handle_interruption()
        self._chunks = 0

    async def reset(self):   # also called by flush() at the end of each answer
        await super().reset()
        self._chunks = 0


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

    async def run_tts(self, text: str, context_id: str):
        if context_id != self._context_id:   # first chunk of a new answer
            self._context_id = context_id
            if self._user_language() in self._voices:
                self._last_lang = self._user_language()
        lang = detect_language(text, default=self._last_lang)
        if lang not in self._voices:
            lang = "en"
        self._last_lang = lang
        self._voice = self._voices[lang]
        logger.debug(f"{self}: speaking [{lang}] {text}")
        trimming = True   # until the first non-silent audio of this chunk
        async for frame in super().run_tts(text, context_id):
            if trimming and isinstance(frame, TTSAudioRawFrame):
                audio, trimming_done = trim_leading_silence(frame.audio, frame.sample_rate)
                if not audio:
                    continue   # all silence: drop the frame
                frame.audio, frame.num_frames = audio, len(audio) // (2 * frame.num_channels)
                trimming = not trimming_done
            yield frame


BilingualPiperTTSService = MultilingualPiperTTSService  # old name still works
