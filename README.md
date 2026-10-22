# Skyy – local voice AI concierge

Skyy is a voice concierge for a college. You speak to it in English, Spanish or French, and it answers out loud in the same language. Everything runs on campus hardware, so no audio or text goes out to the internet.

| Device | Role |
|---|---|
| **Laptop** (Lenovo Aura; an HP Spectre x360 for now) | Runs the voice loop: microphone → speech detection → Whisper (speech to text) → LLM call → Piper (text to speech) → speaker. Built with [Pipecat](https://github.com/pipecat-ai/pipecat). |
| **Jetson AGX Orin 64GB** | AI server: the Gemma LLMs, EmbeddingGemma, Laya and the Qdrant knowledge base. |

The two devices talk over a USB-C cable (or the local network). The Jetson is `192.168.55.1` and the laptop is `192.168.55.100`.

---

## Part 1 – Jetson AGX Orin server

### What runs on the Jetson

| Service | Model / software | Port | Purpose |
|---|---|---|---|
| Main LLM | **Gemma 4 26B-A4B** (instruct, Q4_K_M GGUF) via llama.cpp | 8080 | Writes Skyy's answers. A mixture-of-experts model: 26B parameters in total, about 4B active per token, so it's fast for its size. |
| Fallback LLM | **Gemma 4 E4B** (instruct, Q4 GGUF) via llama.cpp | 8080 (instead of the 26B) | Smaller and faster. Use it if the 26B is too slow or memory is tight. |
| Embeddings | **EmbeddingGemma 2** (Unsloth GGUF `UD-Q4_K_XL`) via llama.cpp | 8082 | Turns text into vectors for knowledge-base search. Multilingual, so a Spanish question can match an English page. Needs a llama.cpp build with EmbeddingGemma 2 support (ggml-org/llama.cpp PR #30054). |
| Decision model | **Laya** (GGUF) via llama.cpp | 8081 | A fast "System 1" model: picks one option from a list (which skill should answer?) or gives a yes/no score (is this answer grounded?) in tens of milliseconds. Endpoint `/v1/systemone`. |
| Vector database | **Qdrant** | 6333 (REST), 6334 (gRPC) | Stores the college knowledge base. Hybrid search: dense (EmbeddingGemma) plus sparse (BM25) vectors. |

> **Why Laya and not Jev:** Jev is a hosted API, so using it would send visitors' questions off campus. Laya has open weights and runs locally on the Jetson.

### 1. Base system

1. Flash **JetPack 7.2** with NVIDIA SDK Manager. It includes CUDA 13.2. Then check it:
   ```bash
   cat /etc/nv_tegra_release
   nvcc --version            # if missing: export PATH=/usr/local/cuda/bin:$PATH
   ```
2. Update packages and install build tools:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential cmake git curl python3-venv python3-pip
   ```
3. Install the monitoring tool (optional but useful): `sudo pip3 install -U jetson-stats`, then run `jtop`.

### 2. Free disk space

The 64 GB eMMC fills up fast, and large containers can fail halfway with "no space left on device". Check before downloading anything:

```bash
df -h /
du -sh ~/.cache/huggingface/hub/* 2>/dev/null | sort -h   # old model downloads
```

- Delete Hugging Face cache folders you no longer need (`rm -rf ~/.cache/huggingface/hub/models--<name>`).
- Remove unused Docker images (`docker image prune -a`). Use llama.cpp built from source rather than multi-GB containers.
- Nsight profiling tools take several GB. **Before** removing them, mark the CUDA libraries as manually installed so `apt autoremove` doesn't take CUDA with them:
  ```bash
  sudo apt-mark manual $(dpkg -l | awk '/^ii/ && /cuda|libcublas|libcudnn|libnpp|libcufft|libcurand|libcusolver|libcusparse/ {print $2}')
  sudo apt remove -s 'nsight-*'      # -s = simulate first; check that no cuda-* package is listed
  sudo apt remove 'nsight-*'
  ```
- An NVMe SSD makes trying other models much easier.

Rough disk budget: Gemma 4 26B-A4B ≈ 16 GB, Gemma 4 E4B ≈ 3–5 GB, EmbeddingGemma ≈ 0.6 GB, Laya ≈ 2 GB, Qdrant plus llama.cpp ≈ 1 GB.

### 3. Performance settings

Run these before every benchmark or demo:

```bash
sudo nvpmodel -m 0        # MAXN: all CPU cores and full GPU power
sudo jetson_clocks        # lock clocks at maximum
sudo nvpmodel -q          # confirm the mode
```

`nvpmodel` survives a reboot; `jetson_clocks` doesn't. To apply it at boot:

```bash
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'EOF'
[Unit]
Description=Max Jetson clocks
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now jetson-clocks
```

Optional: if no monitor is attached, boot without the desktop to free about 1 GB of RAM: `sudo systemctl set-default multi-user.target` (undo with `graphical.target`).

### 4. Build llama.cpp with CUDA

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`87` is the Orin's GPU architecture. The binaries end up in `~/llama.cpp/build/bin/` (`llama-server`, `llama-bench`, …).

Laya uses its own build folder (`build-laya/`), so it doesn't overwrite the main build. Build it the same way with `-B build-laya`, following the instructions on the Laya GGUF model card.

### 5. Download the models

```bash
python3 -m venv ~/skyy-venv && source ~/skyy-venv/bin/activate
pip install -U "huggingface_hub[cli]"
mkdir -p ~/models && cd ~/models

hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --local-dir .
hf download unsloth/gemma-4-E4B-it-GGUF   --include "*Q4_K_M*" --local-dir .
hf download ggml-org/embeddinggemma-300M-GGUF --local-dir .
hf download ggml-org/Laya-GGUF --local-dir .
```

Gemma models require accepting the license on Hugging Face first (`hf auth login`). Check the exact filenames with `ls -lh ~/models` and use them in the commands below.

### 6. Run the servers

Each command runs in its own terminal (or as a service, see step 8).

**Gemma 4 26B-A4B (main LLM)**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --jinja --reasoning-budget 0
```

**Gemma 4 E4B (fallback, same port – stop the 26B first)**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --jinja --reasoning-budget 0
```

**EmbeddingGemma 2**
```bash
hf download unsloth/embeddinggemma-2-GGUF embeddinggemma-2-UD-Q4_K_XL.gguf --local-dir ~/models
~/llama.cpp/build/bin/llama-server \
  -m ~/models/embeddinggemma-2-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8082 -ngl 99 --embeddings --pooling mean --ctx-size 2048
```
EmbeddingGemma 2 uses the same task prefixes as the original model: `task: search result | query: <question>` for questions and `title: none | text: <chunk>` for knowledge-base passages. The knowledge base (`kb.py`) already sends them. The embedding size (768) is read from the model when the Qdrant collection is built, so after switching models re-run `08_kb_index.py` (it recreates the collection) and recalibrate `SKYY_KB_MIN_SCORE`.

The original EmbeddingGemma 300M (`embeddinggemma-300M-Q8_0.gguf`, same port) also works with the same prefixes; use it only as a comparison, since the rest of the setup assumes EmbeddingGemma 2.

**Laya**
```bash
~/llama.cpp/build-laya/bin/llama-server -m ~/models/<laya-file>.gguf --host 0.0.0.0 --port 8081 -ngl 99
```

What the flags do:

| Flag | Why |
|---|---|
| `--host 0.0.0.0` | Listen on the network, not just the Jetson itself. Without it, the laptop can't connect. |
| `-ngl 99` | Put every layer on the GPU. |
| `-c 8192` | Context size. Enough for the system prompt, 3–5 knowledge-base chunks and a short history. Longer contexts slow the Orin down. |
| `--jinja` | Use the model's own chat template (needed for Gemma 4 and tool calling). |
| `--reasoning-budget 0` | Turn off "thinking". Otherwise the model reasons silently before answering, which adds seconds of delay to a voice reply. |

Quick checks from the Jetson:

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8082/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input":"task: search result | query: library hours"}' | head -c 200
~/llama.cpp/build/bin/llama-bench -m ~/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -ngl 99   # tokens/s
```

### 7. Qdrant

Install the ARM64 release binary and run it as a service:

```bash
cd /tmp
curl -LO https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-unknown-linux-musl.tar.gz
tar xzf qdrant-aarch64-unknown-linux-musl.tar.gz
sudo mv qdrant /usr/local/bin/
sudo mkdir -p /var/lib/qdrant/storage /var/lib/qdrant/snapshots
sudo chown -R $USER /var/lib/qdrant
```

`/etc/systemd/system/qdrant.service`:

```ini
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
User=<your-user>
WorkingDirectory=/var/lib/qdrant
Environment=QDRANT__STORAGE__STORAGE_PATH=/var/lib/qdrant/storage
Environment=QDRANT__STORAGE__SNAPSHOTS_PATH=/var/lib/qdrant/snapshots
Environment=QDRANT__SERVICE__API_KEY=<choose-a-key>
Environment=QDRANT__TELEMETRY_DISABLED=true
ExecStart=/usr/local/bin/qdrant
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qdrant
systemctl status qdrant
curl http://localhost:6333/collections                           # 401 = the API key is working
curl -H "api-key: <choose-a-key>" http://localhost:6333/collections   # {"result":{"collections":[]},...}
```

The knowledge base uses one collection with a dense vector (EmbeddingGemma, 768 dimensions, cosine) and a sparse BM25 vector for hybrid search. Python access: `pip install qdrant-client` in `~/skyy-venv`.

### 8. Start everything at boot (optional)

Each llama-server can get its own unit, like Qdrant's. Example for the main LLM, `/etc/systemd/system/gemma.service`:

```ini
[Unit]
Description=Gemma 4 26B-A4B (llama-server :8080)
After=network.target jetson-clocks.service

[Service]
User=<your-user>
ExecStart=/home/<your-user>/llama.cpp/build/bin/llama-server -m /home/<your-user>/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --jinja --reasoning-budget 0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Copy it for EmbeddingGemma (`embeddinggemma.service`, port 8082) and Laya (`laya.service`, port 8081). Then run `sudo systemctl enable --now gemma embeddinggemma laya`.

Memory check with everything running: `jtop` or `tegrastats`. The 26B model plus the others fit comfortably in 64 GB.

### 9. USB-C connection to the laptop

Connect the Jetson's USB-C port to the laptop. The Jetson's USB device mode creates a small network: the **Jetson is `192.168.55.1`**, and the laptop gets `192.168.55.100`.

On Windows, if the link drops or requests fail at random (`ConnectionResetError 10054`, ping timeouts):

1. Device Manager → the Jetson's network adapter (RNDIS) → Properties → Power Management → untick "Allow the computer to turn off this device".
2. Turn off USB selective suspend (run in an Admin PowerShell):
   ```powershell
   powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
   powercfg /setdcvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
   powercfg /setactive SCHEME_CURRENT
   ```
3. Give the adapter a static IP: `192.168.55.100`, mask `255.255.255.0`.
4. Quit VPNs (e.g. PIA) and check that security software isn't blocking the adapter.

Test from the laptop: `curl.exe --noproxy "*" http://192.168.55.1:8080/v1/models`

---

## Part 2 – Laptop voice pipeline (Windows 11)

This is the same code that will run on the Aura. The Spectre (i7-12700H, Iris Xe, **no NPU**) is slower, so treat its numbers as a lower bound. On the Aura, re-run steps 3 and 6 with `NPU`.

### Setup (once)

1. Clone the repo to a short path outside OneDrive, e.g. `git clone https://github.com/fritz007x/Skyy-reboot.git C:\skyy`. Run everything from the repo root: the scripts look for `models\` in the current folder.
2. Settings → Privacy & security → Microphone → turn on **Microphone access** and **Let desktop apps access your microphone**.
3. Update the Intel graphics driver (Intel Driver & Support Assistant). OpenVINO needs a recent one to use the iGPU.
4. Sound settings → your microphone → turn **Audio enhancements** off (Windows noise suppression cuts off the start of words). Set the input volume so `01_mic.py` doesn't report clipping.
5. Run the setup script in PowerShell from that folder:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
   If it installs Python, open a new PowerShell window and run it again. Every `[OK]` in the final check means you're ready.

Later sessions:

```powershell
cd C:\skyy
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python check_setup.py --jetson 192.168.55.1
```

### Python packages (`requirements.txt`)

| Package | Used for |
|---|---|
| `pipecat-ai[silero,openai,local,piper,webrtc,runner]==1.12.0` | The voice pipeline framework (pinned). Extras: Silero VAD, OpenAI-compatible LLM client, local audio (PyAudio), Piper, WebRTC transport and the web runner. |
| `openvino-genai` | Whisper on the Intel CPU, iGPU or NPU |
| `piper-tts` | Text to speech (GPL-3.0; bundles espeak-ng) |
| `silero-vad` | Standalone speech-detection test |
| `openai` | Talks to llama-server's OpenAI-compatible API |
| `numpy`, `scipy`, `sounddevice`, `soundfile` | Audio capture, resampling and file I/O |
| `huggingface_hub` | Model downloads |

Models (downloaded to `models\`): **Whisper large-v3-turbo int8** (OpenVINO) and the Piper voices **en_US-lessac-medium**, **es_MX-claude-high** and **fr_FR-siwis-medium**.

### Scripts

Run them in order: each step checks one piece before the full loop.

| Script | What it does |
|---|---|
| `setup.ps1` | One-time setup: installs Python 3.12 with winget if needed, creates `.venv`, installs requirements, downloads models and runs `check_setup.py`. |
| `check_setup.py` | Checks Python version, packages, OpenVINO GPU, audio devices, models and Piper. `--jetson <ip>` also checks the LLM server. |
| `download_models.py` | Downloads Whisper and the Piper voices. `--small` adds Whisper small; `--voices <names>` adds more voices. Skips files already downloaded. |
| `01_mic.py` | Lists audio devices, records a few seconds, reports level and clipping, plays it back. `--save samples/en1.wav` records test clips. |
| `02_vad.py` | Silero VAD on the live mic or a file: prints when speech starts and ends, for tuning the silence wait (`--silence-ms`). |
| `03_whisper.py` | Benchmarks Whisper on CPU/GPU/NPU: latency, real-time factor, transcript and detected language. `--lang es` forces a language. |
| `04_tts.py` | Piper in English, Spanish and French: time to first audio per sentence, with playback. `--en/--es/--fr` compare voices. |
| `05_llm.py` | Gemma on the Jetson: time to first token and tokens/s for EN/ES/FR prompts. Warns if "thinking" is on. |
| `diag_stream.py` | Network diagnostic: repeats streaming requests over fresh connections to spot a flaky link. |
| `06_voice_loop.py` | **Full voice loop**, laptop mic and speakers (PyAudio). Half-duplex by default (mic ignored while Skyy talks, so it can't hear itself); `--barge-in` with headphones. |
| `07_voice_webrtc.py` | **Full voice loop through the browser** (http://localhost:7860). The browser does echo cancellation, noise suppression and auto gain, so it works on speakers and allows interrupting. Configured with `SKYY_*` environment variables. This is the version to build on. Answers from the MDC FAQ knowledge base (`SKYY_KB_MODE`, see `kb.py`). |
| `08_kb_index.py` | Loads the MDC FAQ (`mdc_faq_kb.jsonl`) into Qdrant: EmbeddingGemma vectors for every entry, then test searches with timings and scores (to pick `SKYY_KB_MIN_SCORE`). `--test` only searches. Needs `SKYY_QDRANT_KEY`. |
| `09_kb_latency.py` | Compares the knowledge-base modes (`off`, `prompt`, `rag`): time from question to the LLM's first token, retrieval included, and how many prompt tokens llama-server had to process. |

Shared modules used by steps 6 and 7:

| Module | What it does |
|---|---|
| `ov_whisper_stt.py` | Pipecat speech-to-text service running Whisper on OpenVINO. Limits languages to EN/ES/FR: if Whisper guesses another language (e.g. Urdu for short Spanish), it re-transcribes in the allowed languages and keeps the most confident result. Uses "Sky" as a hint so the wake word isn't heard as "Hi". `SKYY_SAVE_AUDIO=1` saves each clip to `debug_audio/`. |
| `bilingual_tts.py` | Pipecat Piper service with one voice per language. Detects the language of each sentence and speaks it with the matching voice. |
| `wake.py` | The "Sky" wake word. While asleep, speech without "Sky" is ignored. After "Sky", Skyy stays awake for 20 s after the last exchange. Includes common Whisper spellings (Skye, Skype, …). |
| `kb.py` | The MDC FAQ knowledge base. `SKYY_KB_MODE=prompt` (default): the whole FAQ (~2.5k tokens) goes in the system prompt, which llama-server caches, so no per-turn retrieval. `rag`: `KnowledgeInjector` embeds each question with EmbeddingGemma, searches Qdrant and adds the top entries to the question. `warm_up()` caches the system prompt at startup. |
| `skyy_processors.py` | `WakeWordJoiner`: if you say "Sky", pause, then ask, both parts are joined into one question instead of a greeting plus an interruption. `HistoryCap`: the LLM only sees the last 6 messages, so old or misheard exchanges don't pile up. |

### Run in order

| Step | Command | Pass when |
|---|---|---|
| 1 Mic | `python 01_mic.py --list` then `python 01_mic.py` | Level "fine", no clipping, playback clear |
| 1b Clips | `python 01_mic.py --save samples/en1.wav` (2–3 each in English, Spanish, French) | Clips saved |
| 2 VAD | `python 02_vad.py` (try `--silence-ms 600`) | Start/end match your speech |
| 3 Whisper | `python 03_whisper.py samples/*.wav` | Correct text and language |
| 4 TTS | `python 04_tts.py` | First audio well under 1 s; right voice per language |
| 5 LLM | `python 05_llm.py --host 192.168.55.1` | First token under ~1 s, no thinking warning |
| 6 Loop | `python 06_voice_loop.py --host 192.168.55.1` | You say "Sky, …" and Skyy answers out loud |
| 7 Browser loop | `$env:SKYY_JETSON="192.168.55.1"; python 07_voice_webrtc.py`, then open http://localhost:7860 in Chrome/Edge and click Connect | Works on speakers without Skyy answering itself |

Step 7 options: `SKYY_HALF_DUPLEX=1` (no interrupting), `SKYY_NO_WAKE=1`, `SKYY_WAKE_TIMEOUT`, `SKYY_WAKE_WORDS`, `SKYY_STOP_SECS`, `SKYY_HISTORY`, `SKYY_STT_DEVICE`, `SKYY_VOICE_EN/ES/FR`. Don't pass `--host` to step 7; that sets the web server's own address.

Target for the demo: **under ~3 s** from the end of your sentence to Skyy's first word (the `user -> bot latency` line in the log).

### Record your results

| Measurement | CPU | iGPU | Notes |
|---|---|---|---|
| Whisper, 5 s English clip (ms) | | | |
| Whisper, 5 s Spanish clip (ms) | | | language detected right? |
| Piper first audio, EN / ES / FR (ms) | | – | |
| LLM first token / tok/s | – | – | over USB-C |
| Loop: user → bot latency (s) | | | steps 6 and 7 |

### Troubleshooting

- **Step 5 can't connect** → llama-server isn't started with `--host 0.0.0.0`, or the USB-C link is down (see Part 1, step 9). Test with `curl.exe --noproxy "*" http://192.168.55.1:8080/v1/models`.
- **Random "connection forcibly closed" errors** → flaky USB link: adapter power management, USB selective suspend, static IP, VPN (Part 1, step 9).
- **Skyy ignores "Sky"** → check how Whisper wrote it in the `Heard [...]` line and add that spelling: `--wake-words sky,skye,<new one>` (step 6) or `$env:SKYY_WAKE_WORDS` (step 7).
- **Skyy answers itself** (its own words show up as `Heard [...]`) → don't use `--barge-in` on speakers; use headphones or step 7.
- **Cuts you off mid-sentence** → raise `--stop-secs` / `SKYY_STOP_SECS` (0.8–1.0).
- **Spanish or French read with the English voice** → check the log for `speaking [es]` / `speaking [fr]`. If a sentence is detected wrong, its words may be missing from the lists in `bilingual_tts.py`.
- **Wrong language transcribed** → record the clip with `SKYY_SAVE_AUDIO=1` and test it with `03_whisper.py --lang es`.
- **Wrong mic in step 6** → PyAudio numbers devices differently from step 1. List them with
  `python -c "import pyaudio;p=pyaudio.PyAudio();[print(i,p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`
  and pass `--in-device N --out-device M`.
- **Windows shows no audio devices** → restart the audio services (Admin PowerShell: `Restart-Service Audiosrv,AudioEndpointBuilder -Force`), re-enable disabled devices in `mmsys.cpl`, or reboot.
- **pip fails building PyAudio or aiortc** → you're not on Python 3.12 64-bit; `python check_setup.py` shows the version.
- **"no GPU device"** → update the Intel driver and reboot; use `--devices CPU` / `--stt-device CPU` meanwhile.
- **Want another Piper voice** → `python download_models.py --voices en_US-amy-medium`; browse voices at https://huggingface.co/rhasspy/piper-voices.
