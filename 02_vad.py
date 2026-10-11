"""Step 2 - Silero VAD: does it catch start/end of speech without cutting you off?

  python 02_vad.py                         # live mic, Ctrl+C to stop
  python 02_vad.py --silence-ms 500        # wait longer before calling end of speech
  python 02_vad.py --file samples/en1.wav  # run on a recorded clip

Try: one sentence, a pause mid-sentence ("Where is... the library?"), background noise.
Pipecat's default end-of-speech wait is 0.2 s; Skyy will likely want 0.5-0.8 s.
"""
import argparse, queue, time
import numpy as np
from silero_vad import load_silero_vad, VADIterator

SR, CHUNK = 16000, 512  # Silero needs 512-sample chunks at 16 kHz (32 ms)

ap = argparse.ArgumentParser()
ap.add_argument("--file")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--silence-ms", type=int, default=300, help="silence before 'end'")
ap.add_argument("--device", type=int)
a = ap.parse_args()

model = load_silero_vad(onnx=True)
vad = VADIterator(model, threshold=a.threshold, sampling_rate=SR,
                  min_silence_duration_ms=a.silence_ms, speech_pad_ms=30)


def handle(chunk, t):
    ev = vad(chunk, return_seconds=True)
    if ev and "start" in ev:
        print(f"[{t:6.2f}s] speech START")
    elif ev and "end" in ev:
        print(f"[{t:6.2f}s] speech END")


if a.file:
    import soundfile as sf
    audio, sr = sf.read(a.file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{a.file} is {sr} Hz; record with 01_mic.py (16 kHz)"
    t0 = time.perf_counter()
    for i in range(0, len(audio) - CHUNK, CHUNK):
        handle(audio[i:i + CHUNK], i / SR)
    n = len(audio) // CHUNK
    print(f"VAD cost: {(time.perf_counter() - t0) / n * 1000:.2f} ms per 32 ms chunk")
else:
    import sounddevice as sd
    q = queue.Queue()
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                            device=a.device, callback=lambda d, f, ti, s: q.put(d[:, 0].copy()))
    print("Listening... speak, pause, speak again. Ctrl+C to stop.")
    t, cost = 0.0, []
    with stream:
        try:
            while True:
                c = q.get()
                s = time.perf_counter()
                handle(c, t)
                cost.append(time.perf_counter() - s)
                t += CHUNK / SR
        except KeyboardInterrupt:
            pass
    if cost:
        print(f"\nVAD cost: {np.mean(cost) * 1000:.2f} ms per 32 ms chunk")
