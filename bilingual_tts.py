"""Piper TTS that picks the English, Spanish or French voice for each sentence.

Piper voices are one language each, so a single PiperTTSService reads Spanish or
French answers with the English voice. This service loads one voice per language
and, for every sentence Pipecat sends it, detects the language of that sentence's
text and speaks it with the matching voice. It looks at what Skyy is about to SAY
(the LLM's answer), so it works even when Whisper doesn't report a language.

detect_language() is also used by ov_whisper_stt.py to label transcripts.
"""
import re
from pathlib import Path

from loguru import logger
from piper import PiperVoice
from piper.download_voices import download_voice

from pipecat.services.piper.tts import PiperTTSService

# Common words that are distinctive for each language. Words shared by two of them
# ("la", "de", "en", "que", "a" ...) are left out on purpose: they don't help decide.
WORDS = {
    "en": set("""the an and or of to in is are was were be it this that these those for with
        on at by from as but not you your we our they their he she his her my can could would
        will what where when how which who hello hi thanks thank open until library building
        floor class classes help room today there here""".split()),
    "es": set("""el los las un una unos unas del al y o es está están son para por con sin
        como pero más muy sí no hay su sus mi tu yo usted ustedes nosotros este esta estos
        estas ese esa eso aquí allí dónde donde cuándo cuando qué cómo cuál puede puedo tiene
        tengo hola gracias buenos buenas días tardes noches abierto abierta hasta desde
        biblioteca edificio piso clase clases ayuda ayudar horario""".split()),
    "fr": set("""le les des du au aux et est sont un une je tu il elle nous vous ils elles
        pour avec sans sur dans par mais pas ne plus très oui non ce cette ces c'est qui où
        quand comment quel quelle peut peux avez avons bonjour bonsoir merci salut ouvert
        ouverte jusqu bibliothèque bâtiment étage cours aide aider horaires aujourd'hui""".split()),
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


class MultilingualPiperTTSService(PiperTTSService):
    def __init__(self, *, voices: dict, download_dir: Path = Path("models/piper"), **kwargs):
        """voices: {"en": "en_US-lessac-medium", "es": "es_MX-claude-high", "fr": "fr_FR-siwis-medium"}"""
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

    async def run_tts(self, text: str, context_id: str):
        lang = detect_language(text, default=self._last_lang)
        if lang not in self._voices:
            lang = "en"
        self._last_lang = lang
        self._voice = self._voices[lang]
        logger.debug(f"{self}: speaking [{lang}] {text}")
        async for frame in super().run_tts(text, context_id):
            yield frame


BilingualPiperTTSService = MultilingualPiperTTSService  # old name still works
