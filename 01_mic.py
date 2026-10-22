"""Step 1 - microphone and speaker check, and recording test clips.

  python 01_mic.py --list                       # show audio devices
  python 01_mic.py                              # record 5 s, report level, play back
  python 01_mic.py --save samples/en1.wav       # record a clip for the Whisper test
  python 01_mic.py --save samples/es1.wav --seconds 6 --device 3
"""
import argparse, os
import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000  # Whisper and Silero both want 16 kHz mono

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--seconds", type=float, default=5)
ap.add_argument("--device", type=int, help="input device index from --list")
ap.add_argument("--save", help="path of the WAV to write")
ap.add_argument("--no-play", action="store_true")
a = ap.parse_args()

if a.list:
    print(sd.query_devices())
    print("\nDefault (input, output):", sd.default.device)
    raise SystemExit

input(f"Press Enter, then speak for {a.seconds:.0f} seconds...")
audio = sd.rec(int(a.seconds * SR), samplerate=SR, channels=1, dtype="float32", device=a.device)
sd.wait()
audio = audio[:, 0]

peak = float(np.max(np.abs(audio)))
rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)
print(f"Peak {peak:.2f}   RMS {rms_db:.1f} dBFS")
if peak < 0.05:
    print("-> Very quiet. Check the input device (--list / --device) or Windows mic level.")
elif peak > 0.98:
    print("-> Clipping. Lower the mic level in Windows Sound settings.")
else:
    print("-> Level looks fine.")

if a.save:
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    sf.write(a.save, audio, SR)
    print(f"Saved {a.save}")

if not a.no_play:
    print("Playing back...")
    sd.play(audio, SR)
    sd.wait()
