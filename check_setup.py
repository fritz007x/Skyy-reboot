"""Check that everything is installed before running the test scripts.

  python check_setup.py                      # local checks
  python check_setup.py --jetson 192.168.1.50   # also check the Jetson LLM
"""
import argparse, importlib, os, platform, sys

ap = argparse.ArgumentParser()
ap.add_argument("--jetson")
ap.add_argument("--port", type=int, default=8080)
a = ap.parse_args()

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"  [OK]   {name}{': ' + msg if msg else ''}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


print(f"Python {platform.python_version()} ({platform.architecture()[0]}) on {platform.platform()}")
if sys.version_info[:2] not in ((3, 12), (3, 13)):
    print("  [WARN] Python 3.13 is recommended (3.12 also works).")
if sys.prefix == sys.base_prefix:
    print("  [WARN] Not inside the .venv - run .\\.venv\\Scripts\\Activate.ps1 first.")

print("\nPackages")
for mod in ["numpy", "scipy", "sounddevice", "soundfile", "huggingface_hub", "openvino",
            "openvino_genai", "piper", "silero_vad", "openai", "pipecat", "pyaudio",
            "aiortc", "fastapi", "uvicorn"]:
    check(mod, lambda m=mod: getattr(importlib.import_module(m), "__version__", ""))

print("\nOpenVINO devices (Whisper)")


def ov_devices():
    import openvino as ov
    core = ov.Core()
    devs = core.available_devices
    names = [f"{d} = {core.get_property(d, 'FULL_DEVICE_NAME')}" for d in devs]
    if not any(d.startswith("GPU") for d in devs):
        raise RuntimeError("no GPU device. Update the Intel graphics driver. Found: " + ", ".join(names))
    if not any(d.startswith("NPU") for d in devs):
        names.append("no NPU (install the Intel NPU driver on the Aura)")
    return "; ".join(names)


check("CPU + GPU (+ NPU) visible", ov_devices)

print("\nAudio devices")


def sd_devices():
    import sounddevice as sd
    i, o = sd.default.device
    return f"default in = {sd.query_devices(i)['name']} | default out = {sd.query_devices(o)['name']}"


def pyaudio_devices():
    import pyaudio
    p = pyaudio.PyAudio()
    n = sum(1 for k in range(p.get_device_count()) if p.get_device_info_by_index(k)["maxInputChannels"] > 0)
    d = p.get_default_input_device_info()["name"]
    p.terminate()
    if n == 0:
        raise RuntimeError("no input devices: check Windows mic privacy settings")
    return f"{n} inputs, default = {d}"


check("sounddevice (steps 1-4)", sd_devices)
check("PyAudio (step 6)", pyaudio_devices)

print("\nModels")


def exists(path, what):
    def f():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run python download_models.py")
        return what
    return f


check("Whisper large-v3-turbo (OpenVINO)",
      exists("models/whisper-large-v3-turbo-int8-ov/openvino_encoder_model.xml", "found"))
for _v in ("en_US-lessac-medium", "es_MX-claude-high", "fr_FR-siwis-medium"):
    check(f"Piper voice {_v}", exists(f"models/piper/{_v}.onnx", "found"))


def silero():
    from silero_vad import load_silero_vad
    load_silero_vad(onnx=True)
    return "loads"


check("Silero VAD", silero)


def piper_speaks():
    from piper import PiperVoice
    v = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
    n = sum(len(c.audio_float_array) for c in v.synthesize("Hello."))
    if n == 0:
        raise RuntimeError("produced no audio")
    return f"synthesized {n / v.config.sample_rate:.1f} s of audio"


check("Piper speaks", piper_speaks)

if a.jetson:
    print("\nJetson")

    def llm():
        from openai import OpenAI
        c = OpenAI(base_url=f"http://{a.jetson}:{a.port}/v1", api_key="none", timeout=10,
                   default_headers={"Connection": "close"})
        return "model " + c.models.list().data[0].id

    check(f"LLM at {a.jetson}:{a.port}", llm)

print("\nAll good - start with python 01_mic.py" if ok else
      "\nFix the [FAIL] lines above (see README > Troubleshooting), then run this again.")
