"""Step 4 - Piper TTS: how fast does speech start, in English, Spanish and French?

  python 04_tts.py                                   # default English, Spanish and French voices
  python 04_tts.py --en en_US-lessac-high --es es_MX-ald-medium   # compare other voices
  python 04_tts.py --no-play                         # numbers only

Piper uses one model per voice/language. "First audio" is what matters for voice:
Pipecat sends text to TTS sentence by sentence, so the listener waits for the
first sentence, not the whole answer.
Voices: https://huggingface.co/rhasspy/piper-voices (x_low/low < medium < high = slower, nicer)
"""
import argparse, os, time
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
