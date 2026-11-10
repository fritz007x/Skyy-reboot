"""Skyy TTS server on the Jetson: one cloned voice (XTTS-v2) for English, Spanish and French.

The laptop has no CUDA GPU, so XTTS runs here, next to the llama-servers, and the voice loop
streams from it over HTTP (bilingual_tts.MultilingualXTTSService).

Each folder in XTTS_VOICES_DIR is a voice: its .wav clips (mono, 6-15 s each, clean speech,
no music; check them with 11_voice_ref.py on the laptop) are turned into XTTS conditioning
latents once at startup, so a request only pays for synthesis.

  POST /tts_stream  {"text": "...", "language": "en"|"es"|"fr", "voice": "jabberjaw"}
                    -> raw 16-bit mono PCM at 24 kHz, streamed as XTTS generates it
  GET  /health      -> {"voices": [...], "languages": [...], "sample_rate": 24000}

Setup (once, see README "Part 1 > 10. Cloned voice"):
  python3 -m venv ~/skyy-tts-venv && source ~/skyy-tts-venv/bin/activate
  pip install <PyTorch wheel for your JetPack>   # NVIDIA's Jetson wheels, not PyPI's torch
  pip install coqui-tts fastapi uvicorn
  huggingface-cli download coqui/XTTS-v2 --local-dir ~/models/xtts_v2

Run by hand:  COQUI_TOS_AGREED=1 python tts_server.py   (as a service: install_services.sh --tts)

Environment:
  XTTS_MODEL_DIR    ~/models/xtts_v2      config.json, model.pth, vocab.json
  XTTS_VOICES_DIR   ~/voices              one sub-folder of .wav clips per voice
  XTTS_PORT         8083
  XTTS_TEMPERATURE  0.65                  lower = steadier, higher = more expressive
  XTTS_SPEED        1.0
  XTTS_CHUNK        20                    GPT tokens per streamed chunk (smaller = sooner first audio)

XTTS-v2 weights are under the Coqui Public Model License: non-commercial use only.
"""
import asyncio
import os
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import wave

import torchaudio.functional as AF
from pydantic import BaseModel
import TTS.tts.models.xtts as xtts_module
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_wav(path, sampling_rate):
    """Read a 16-bit PCM WAV with the standard library. coqui-tts >= 0.27 loads audio through
    torchcodec, which also needs FFmpeg's shared libraries; this keeps the server independent
    of them (11_voice_ref.py checks the clips are 16-bit PCM)."""
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    audio = torch.from_numpy(x.mean(axis=1).astype(np.float32) / 32768).unsqueeze(0)
    return AF.resample(audio, sr, sampling_rate) if sr != sampling_rate else audio


xtts_module.load_audio = load_wav   # used by get_conditioning_latents()

MODEL_DIR = Path(os.path.expanduser(os.environ.get("XTTS_MODEL_DIR", "~/models/xtts_v2")))
VOICES_DIR = Path(os.path.expanduser(os.environ.get("XTTS_VOICES_DIR", "~/voices")))
PORT = int(os.environ.get("XTTS_PORT", "8083"))
TEMPERATURE = float(os.environ.get("XTTS_TEMPERATURE", "0.65"))
SPEED = float(os.environ.get("XTTS_SPEED", "1.0"))
CHUNK = int(os.environ.get("XTTS_CHUNK", "20"))
LANGUAGES = ("en", "es", "fr")   # XTTS knows more; Skyy only speaks these
SAMPLE_RATE = 24000
WARM_UP = {"en": "Hi, I'm Skyy.", "es": "Hola, soy Skyy.", "fr": "Bonjour, je suis Skyy."}


def log(msg):
    print(f"[skyy-tts] {msg}", flush=True)


def load_model() -> Xtts:
    t = time.perf_counter()
    config = XttsConfig()
    config.load_json(str(MODEL_DIR / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(MODEL_DIR), use_deepspeed=False)
    model.cuda().eval()
    log(f"XTTS-v2 loaded from {MODEL_DIR} in {time.perf_counter() - t:.1f} s")
    return model


def load_voices(model: Xtts) -> dict:
    voices = {}
    for folder in sorted(p for p in VOICES_DIR.iterdir() if p.is_dir()):
        clips = sorted(str(p) for p in folder.glob("*.wav"))
        if not clips:
            continue
        latent, embedding = model.get_conditioning_latents(audio_path=clips)
        voices[folder.name] = (latent, embedding)
        log(f"voice '{folder.name}': {len(clips)} clips")
    if not voices:
        raise SystemExit(f"No voices: put .wav clips in {VOICES_DIR}/<voice name>/")
    return voices


def stream(text: str, language: str, voice: tuple):
    """Yield 16-bit PCM byte chunks as XTTS generates them."""
    latent, embedding = voice
    for chunk in model.inference_stream(
            text, language, latent, embedding, stream_chunk_size=CHUNK,
            temperature=TEMPERATURE, speed=SPEED, enable_text_splitting=False):
        audio = chunk.float().clamp(-1, 1).cpu().numpy()
        yield (audio * 32767).astype(np.int16).tobytes()


model = load_model()
VOICES = load_voices(model)
for lang, text in WARM_UP.items():   # first CUDA run is slow; don't make a visitor wait for it
    t = time.perf_counter()
    for _ in stream(text, lang, next(iter(VOICES.values()))):
        pass
    log(f"warm-up [{lang}] {time.perf_counter() - t:.2f} s")

app = FastAPI()
gpu = asyncio.Lock()   # one kiosk, one synthesis at a time


class Speak(BaseModel):
    text: str
    language: str = "en"
    voice: str | None = None


@app.get("/health")
def health():
    return {"voices": list(VOICES), "languages": list(LANGUAGES), "sample_rate": SAMPLE_RATE}


@app.post("/tts_stream")
async def tts_stream(req: Speak, request: Request):
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of {LANGUAGES}")
    voice = req.voice or next(iter(VOICES))
    if voice not in VOICES:
        raise HTTPException(404, f"unknown voice '{voice}'; have {list(VOICES)}")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")

    async def body():
        async with gpu:
            t0 = time.perf_counter()
            first, n = None, 0
            gen = stream(text, req.language, VOICES[voice])
            try:
                while True:
                    # Each chunk is generated in a worker thread so the server stays responsive
                    chunk = await asyncio.to_thread(next, gen, None)
                    if chunk is None:
                        break
                    if await request.is_disconnected():   # barge-in: stop using the GPU
                        log(f"client left, stopped [{req.language}] {text[:40]}")
                        return
                    first = first or time.perf_counter() - t0
                    n += len(chunk)
                    yield chunk
            finally:
                gen.close()
            total = time.perf_counter() - t0
            dur = n / 2 / SAMPLE_RATE
            log(f"[{req.language}] first {first or 0:.2f} s, total {total:.2f} s for {dur:.1f} s "
                f"(RTF {total / dur if dur else 0:.2f}): {text[:60]}")

    return StreamingResponse(body(), media_type="audio/L16; rate=24000; channels=1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
