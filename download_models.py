"""Download every model the tests need into .\\models (run once, ~1 GB).

  python download_models.py                  # Whisper large-v3-turbo + Piper EN/ES/FR voices
  python download_models.py --small          # also Whisper small (faster fallback for the Spectre)
  python download_models.py --voices en_US-lessac-high es_MX-ald-medium   # extra Piper voices
"""

import os
import ssl

# Disable SSL verification for huggingface_hub and urllib/httpx
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

# Patch standard library SSL context directly
ssl._create_default_https_context = ssl._create_unverified_context

import argparse, os
from pathlib import Path
from huggingface_hub import snapshot_download
from piper.download_voices import download_voice

DEFAULT_VOICES = ["en_US-lessac-medium", "es_MX-claude-high", "fr_FR-siwis-medium"]

ap = argparse.ArgumentParser()
ap.add_argument("--small", action="store_true")
ap.add_argument("--voices", nargs="*", default=[])
a = ap.parse_args()

whisper = ["OpenVINO/whisper-large-v3-turbo-int8-ov"]
if a.small:
    whisper.append("OpenVINO/whisper-small-int8-ov")
for repo in whisper:
    dest = os.path.join("models", repo.split("/")[-1])
    print(f"Whisper: {repo} -> {dest}")
    snapshot_download(repo, local_dir=dest)

voice_dir = Path("models/piper")
voice_dir.mkdir(parents=True, exist_ok=True)
for v in DEFAULT_VOICES + a.voices:
    if (voice_dir / f"{v}.onnx").exists() and (voice_dir / f"{v}.onnx.json").exists():
        print(f"Piper: {v} already there")
        continue
    print(f"Piper: downloading {v} ...")
    download_voice(v, voice_dir)

# Silero VAD ships inside the silero-vad and pipecat packages: nothing to download.
print("\nDone. Next: python check_setup.py")
