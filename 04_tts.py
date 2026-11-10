"""Step 4 - Piper TTS: how fast does speech start, in English, Spanish and French?

  python 04_tts.py                                   # default English, Spanish and French voices
  python 04_tts.py --en en_US-lessac-high --es es_MX-ald-medium   # compare other voices
  python 04_tts.py --no-play                         # numbers only
  python 04_tts.py --engine xtts --jetson 192.168.55.1   # the cloned voice on the Jetson (:8083)
  python 04_tts.py --engine xtts --jetson 192.168.55.1 --voice jabberjaw --save samples/xtts

Piper uses one model per voice/language. "First audio" is what matters for voice:
Pipecat sends text to TTS sentence by sentence, so the listener waits for the
first sentence, not the whole answer.
Voices: https://huggingface.co/rhasspy/piper-voices (x_low/low < medium < high = slower, nicer)
"""
import argparse, json, os, time, urllib.request, wave
from pathlib import Path
import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.download_voices import download_voice

ap = argparse.ArgumentParser()
ap.add_argument("--en", default="en_US-lessac-medium")
ap.add_argument("--es", default="es_MX-claude-high")
ap.add_argument("--fr", default="fr_FR-siwis-medium")
ap.add_argument("--no-play", action="store_true")
ap.add_argument("--engine", choices=("piper", "xtts"), default="piper")
ap.add_argument("--jetson", default="192.168.55.1", help="Jetson IP (xtts)")
ap.add_argument("--tts-port", type=int, default=8083)
ap.add_argument("--voice", default="jabberjaw", help="XTTS voice folder on the Jetson")
ap.add_argument("--save", help="folder to save each language's audio as a .wav (xtts)")
a = ap.parse_args()

VOICE_DIR = Path("models/piper")
VOICE_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    (a.en, "Hi, I'm Skyy, the college concierge. The library is in Building C, on the second "
           "floor. It's open until nine tonight."),
    (a.es, "Hola, soy Skyy. La biblioteca está en el edificio C, en el segundo piso. "
           "Está abierta hasta las nueve de la noche."),
    (a.fr, "Bonjour, je suis Skyy. La bibliothèque est dans le bâtiment C, au deuxième étage. "
           "Elle est ouverte jusqu'à vingt et une heures."),
]


def load(voice):
    if not (VOICE_DIR / f"{voice}.onnx").exists():
        print(f"Downloading {voice} ...")
        download_voice(voice, VOICE_DIR)
    t = time.perf_counter()
    v = PiperVoice.load(VOICE_DIR / f"{voice}.onnx")
    print(f"Loaded {voice} in {time.perf_counter() - t:.1f} s ({v.config.sample_rate} Hz)")
    return v


def speak(v, text):
    return np.concatenate([c.audio_float_array for c in v.synthesize(text)])


def xtts_sentence(text, lang):
    """POST one sentence to the Jetson; returns (seconds to first audio, int16 samples)."""
    req = urllib.request.Request(
        f"http://{a.jetson}:{a.tts_port}/tts_stream", method="POST",
        data=json.dumps({"text": text, "language": lang, "voice": a.voice}).encode(),
        headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy on the USB link
    t0, first, pcm = time.perf_counter(), None, bytearray()
    with opener.open(req, timeout=30) as r:
        while chunk := r.read1(4096):
            first = first or time.perf_counter() - t0
            pcm += chunk
    return first, np.frombuffer(bytes(pcm[:len(pcm) // 2 * 2]), dtype=np.int16)


if a.engine == "xtts":   # one voice for all three languages, from jetson/tts_server.py
    XTTS_SR = 24000
    for lang, (_, text) in zip(("en", "es", "fr"), TESTS):
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        t0 = time.perf_counter()
        firsts, parts = [], []
        for s in sentences:
            f, audio = xtts_sentence(s, lang)
            firsts.append(f)
            parts.append(audio)
        total = time.perf_counter() - t0
        audio = np.concatenate(parts)
        dur = len(audio) / XTTS_SR
        print(f"[{lang}] {a.voice}: first audio {firsts[0]*1000:5.0f} ms "
              f"(per sentence {', '.join(f'{x*1000:.0f}' for x in firsts)} ms) | total {total:.2f} s "
              f"for {dur:.1f} s speech (RTF {total/dur:.2f})")
        if a.save:
            Path(a.save).mkdir(parents=True, exist_ok=True)
            with wave.open(str(Path(a.save) / f"{a.voice}_{lang}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(XTTS_SR)
                w.writeframes(audio.tobytes())
        if not a.no_play:
            sd.play(audio, XTTS_SR)
            sd.wait()
    raise SystemExit

for voice_name, text in TESTS:
    v = load(voice_name)
    speak(v, "Warm up.")
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    t0 = time.perf_counter()
    first, parts = None, []
    for s in sentences:  # sentence by sentence, like Pipecat
        parts.append(speak(v, s))
        if first is None:
            first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    audio = np.concatenate(parts)
    sr = v.config.sample_rate
    dur = len(audio) / sr
    print(f"  first audio {first*1000:5.0f} ms | total {total:.2f} s for {dur:.1f} s speech "
          f"(RTF {total/dur:.2f})")
    if not a.no_play:
        sd.play(audio, sr)
        sd.wait()
