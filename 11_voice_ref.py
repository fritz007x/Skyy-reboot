"""Step 11 - check the reference clips for a cloned XTTS voice before copying them to the Jetson.

  python 11_voice_ref.py voices/jabberjaw
  python 11_voice_ref.py voices/jabberjaw --play      # also listen to each clip

XTTS-v2 copies whatever is in the clips: music, laughter, room echo and a second speaker end
up in Skyy's voice. Good clips are one speaker, clean speech, 6-15 s each, 30-60 s in total,
mono WAV at 22.05 or 24 kHz.

Getting clean clips from a recording with music (once, in any Python env):
  pip install demucs
  demucs --two-stems=vocals episode.wav                 # -> separated/htdemucs/episode/vocals.wav
  ffmpeg -i vocals.wav -ss 00:01:12 -t 9 -ac 1 -ar 24000 -af loudnorm voices/jabberjaw/01.wav

Then copy the folder to the Jetson (one folder per voice; the folder name is the voice name):
  scp -r voices/jabberjaw <user>@192.168.55.1:~/voices/
  ssh <user>@192.168.55.1 sudo systemctl restart skyy-tts

Only clone a voice you have the right to use (see README "Cloned voice").
"""
import argparse
import wave
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--play", action="store_true")
a = ap.parse_args()

clips = sorted(a.folder.glob("*.wav"))
if not clips:
    raise SystemExit(f"No .wav files in {a.folder}")

total, problems = 0.0, 0
for path in clips:
    try:
        with wave.open(str(path)) as w:
            sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
    except wave.Error as e:
        print(f"[FAIL] {path.name}: not a PCM WAV ({e}); convert with ffmpeg -ac 1 -ar 24000")
        problems += 1
        continue
    if width != 2:
        print(f"[FAIL] {path.name}: {8 * width}-bit; use 16-bit (ffmpeg -sample_fmt s16)")
        problems += 1
        continue
    audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)[:, 0].astype(np.float32) / 32768
    dur = len(audio) / sr
    total += dur
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    clipped = float(np.mean(np.abs(audio) > 0.99)) * 100
    # Share of 50 ms frames that are near-silent: long pauses waste reference time
    frames = audio[: len(audio) // (sr // 20) * (sr // 20)].reshape(-1, sr // 20)
    silent = float(np.mean(np.sqrt((frames ** 2).mean(axis=1)) < 0.01)) * 100 if len(frames) else 0

    notes = []
    if ch != 1:
        notes.append(f"{ch} channels (use mono: -ac 1)")
    if sr not in (22050, 24000):
        notes.append(f"{sr} Hz (use 24000: -ar 24000)")
    if dur < 4:
        notes.append("shorter than 4 s")
    elif dur > 20:
        notes.append("longer than 20 s (cut it in two)")
    if clipped > 0.1:
        notes.append(f"{clipped:.1f}% clipped")
    if rms_db < -30:
        notes.append(f"quiet ({rms_db:.0f} dBFS; use -af loudnorm)")
    if silent > 30:
        notes.append(f"{silent:.0f}% silence (trim pauses)")
    problems += bool(notes)
    print(f"[{'WARN' if notes else ' OK '}] {path.name}: {dur:4.1f} s, {sr} Hz, peak {peak:.2f}, "
          f"{rms_db:.0f} dBFS{' - ' + '; '.join(notes) if notes else ''}")
    if a.play:
        import sounddevice as sd
        sd.play(audio, sr)
        sd.wait()

print(f"\n{len(clips)} clips, {total:.0f} s in total "
      f"({'aim for 30-60 s' if not 30 <= total <= 90 else 'good amount'}).")
print("Ready to copy to the Jetson." if not problems and total >= 10 else
      "Fix the lines above, then run this again.")
