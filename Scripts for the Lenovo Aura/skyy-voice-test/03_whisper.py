"""Step 3 - Whisper with OpenVINO: CPU vs Intel iGPU (and NPU on the Aura).

  python 03_whisper.py samples/en1.wav samples/es1.wav
  python 03_whisper.py samples/*.wav --model OpenVINO/whisper-small-int8-ov
  python 03_whisper.py samples/*.wav --devices GPU --runs 5
  python 03_whisper.py samples/es*.wav --devices GPU --runs 1 --lang es   # force Spanish

Reports load time, median latency per clip, real-time factor (RTF < 1 = faster
than real time), the transcript and the detected language.
The Spectre has no NPU; the same script tries "NPU" automatically on the Aura.
Run it twice on the Aura: the first NPU load compiles the model (slow), the second uses ov_cache/.
"""
import argparse, glob, os, statistics, time
import numpy as np
import soundfile as sf
import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("wavs", nargs="+")
ap.add_argument("--model", default="OpenVINO/whisper-large-v3-turbo-int8-ov",
                help="any OpenVINO/whisper-*-ov repo on Hugging Face")
ap.add_argument("--devices", nargs="*", help="default: every available CPU/GPU/NPU")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--lang", help="force a language (en, es, fr) instead of auto-detect")
a = ap.parse_args()

wavs = [w for p in a.wavs for w in sorted(glob.glob(p))]  # Windows shells don't expand *
model_dir = os.path.join("models", a.model.split("/")[-1])
if not os.path.isdir(model_dir):
    print(f"Downloading {a.model} ...")
    snapshot_download(a.model, local_dir=model_dir)

available = ov.Core().available_devices
print("OpenVINO devices:", available)
devices = a.devices or [d for d in ("CPU", "GPU", "NPU") if any(x.startswith(d) for x in available)]


def load_16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


clips = {w: load_16k(w) for w in wavs}
rows = []
for dev in devices:
    print(f"\n=== {dev} ===")
    try:
        t = time.perf_counter()
        try:  # NPU: 1st load compiles, later loads use ov_cache/
            pipe = ov_genai.WhisperPipeline(model_dir, dev, CACHE_DIR="ov_cache")
        except TypeError:  # older OpenVINO without keyword properties
            pipe = ov_genai.WhisperPipeline(model_dir, dev)
        load = time.perf_counter() - t
    except Exception as e:  # e.g. NPU driver missing
        print(f"  could not load on {dev}: {e}")
        continue
    print(f"  load (incl. compile): {load:.1f} s")
    pipe.generate(next(iter(clips.values())).tolist())  # warm-up
    for path, audio in clips.items():
        times = []
        for _ in range(a.runs):
            t = time.perf_counter()
            kw = {"language": f"<|{a.lang}|>"} if a.lang else {}   # default: auto-detect
            res = pipe.generate(audio.tolist(), task="transcribe", **kw)
            times.append(time.perf_counter() - t)
        dur = len(audio) / 16000
        med = statistics.median(times)
        langs = getattr(res, "languages", None) or ["?"]
        lang = str(langs[0]).strip("<|>")
        text = res.texts[0].strip()
        print(f"  {os.path.basename(path)}: {med*1000:.0f} ms for {dur:.1f} s audio "
              f"(RTF {med/dur:.2f})  lang={lang}\n     \"{text}\"")
        rows.append((dev, os.path.basename(path), med, med / dur, lang))

if rows:
    print("\nSummary (median latency ms / RTF / language)")
    for dev, name, med, rtf, lang in rows:
        print(f"  {dev:4s} {name:20s} {med*1000:7.0f}  {rtf:5.2f}  {lang}")
