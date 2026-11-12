# Skyy – local voice AI concierge

Skyy is a voice concierge for a college. You speak to it in English, Spanish or French, and it answers out loud in the same language. Everything runs on campus hardware, so no audio or text goes out to the internet.

| Device | Role |
|---|---|
| **Laptop** (Lenovo Aura) | Runs the voice loop: microphone → speech detection → Whisper (speech to text) → LLM call → Piper (text to speech) → speaker. Built with [Pipecat](https://github.com/pipecat-ai/pipecat). |
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
| Cloned voice (optional) | **XTTS-v2** (Coqui, maintained as `coqui-tts`) in `jetson/tts_server.py` | 8083 | Speaks English, Spanish and French in one cloned voice (`SKYY_TTS=xtts`). Without it, the laptop speaks with Piper, one voice per language. See step 10. |
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
- Add an NVMe SSD (see below).

#### NVMe SSD

Keep booting from the eMMC and put the large folders on the SSD, with a symlink at each old path, so the services and `/etc/skyy/*.env` keep their paths. The SSD goes in the dev kit's M.2 **Key M** slot (underside; the other slot holds the Wi-Fi card). It must be NVMe, not M.2 SATA. Power off and unplug before fitting it. If `lspci` doesn't list it, reseat it and fit the retaining screw.

1. Find it: `lsblk -o NAME,SIZE,MODEL,FSTYPE,MOUNTPOINT` should show `nvme0n1` with no partitions. **The next step erases it.**
2. Partition, format and mount it at every boot (`nofail`: the Jetson still boots if the SSD dies):
   ```bash
   sudo parted -s /dev/nvme0n1 mklabel gpt mkpart ssd ext4 0% 100%
   sudo mkfs.ext4 -L ssd /dev/nvme0n1p1
   sudo mkdir -p /ssd
   echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p1) /ssd ext4 defaults,noatime,nofail 0 2" | sudo tee -a /etc/fstab
   sudo systemctl daemon-reload && sudo mount -a && sudo chown $USER:$USER /ssd
   ```
3. Move the large folders and leave symlinks (a folder that doesn't exist yet gets an empty one on the SSD):
   ```bash
   sudo systemctl stop 'skyy-*'
   mkdir -p ~/.cache
   for d in models voices skyy-tts-venv .cache/huggingface; do
     if [ -d ~/$d ] && [ ! -L ~/$d ]; then
       mkdir -p "/ssd/$(dirname $d)"
       rsync -a --info=progress2 ~/$d/ /ssd/$d/ && mv ~/$d ~/$d.old
     fi
     mkdir -p /ssd/$d
     [ -e ~/$d ] || ln -s /ssd/$d ~/$d
   done
   df -h ~/models/ ~/voices/ ~/skyy-tts-venv/ ~/.cache/huggingface/   # each should be /dev/nvme0n1p1 on /ssd
   ```
   `~` itself stays on the eMMC: `df -h ~` still shows it, and that's expected.
4. Start the services and check them (`sudo systemctl start skyy-llm skyy-embed`, then `curl localhost:8080/v1/models` and `:8082`). Only then delete the old copies: `rm -rf ~/*.old ~/.cache/*.old`.
5. Large pip installs (for example torch in step 10) also use `~/.cache/pip` and `/tmp` on the eMMC. Point them at the SSD: `echo 'export TMPDIR=/ssd/tmp PIP_CACHE_DIR=/ssd/pip-cache' >> ~/.bashrc`, then `mkdir -p /ssd/tmp /ssd/pip-cache`.

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

Each command runs in its own terminal. To keep them running, use the services in step 8.

**Gemma 4 26B-A4B (main LLM)**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --parallel 1 --jinja --reasoning-budget 0
```

**Gemma 4 E4B (fallback, same port – stop the 26B first)**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --parallel 1 --jinja --reasoning-budget 0
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
| `--parallel 1` | One request slot. Skyy holds one conversation at a time, so the whole context belongs to it and the cached system prompt is always reused. With the default 4 slots, quick back-to-back turns that interrupt each other can leave two slots holding the ~4.2k-token prompt, and the next request fails with "Context size has been exceeded". |
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

### 8. Run the llama-servers as services

The units in `jetson/systemd/` run each llama-server under systemd: they start at boot and restart 5 s after a crash (`Restart=on-failure`). If one fails 5 times within 5 minutes, for example because the model path is wrong, systemd stops retrying. Copy the `jetson/` folder to the Jetson and run:

```bash
sudo ./jetson/install_services.sh          # main LLM (:8080) and EmbeddingGemma (:8082)
sudo ./jetson/install_services.sh --laya   # also Laya (:8081)
sudo ./jetson/install_services.sh --tts    # also the cloned voice (:8083, set it up first: step 10)
```

Stop any llama-server you started by hand first, or the services can't get their ports. The script fills in your user and home folder, puts each model path in `/etc/skyy/<name>.env` (`llm`, `embed`, `laya`), and enables the services. Re-running it keeps existing `.env` files. For Laya, set the real filename in `/etc/skyy/laya.env` first.

| Service | Port | Model file |
|---|---|---|
| `skyy-llm` | 8080 | `/etc/skyy/llm.env` |
| `skyy-embed` | 8082 | `/etc/skyy/embed.env` |
| `skyy-laya` | 8081 | `/etc/skyy/laya.env` |
| `skyy-tts` | 8083 | `/etc/skyy/tts.env` (`XTTS_MODEL_DIR`, `XTTS_VOICES_DIR`) |

```bash
systemctl status skyy-llm skyy-embed
journalctl -u skyy-llm -f                  # logs; the 26B takes a while to load
sudo systemctl restart skyy-llm
```

To switch to the E4B fallback, set `MODEL=` in `/etc/skyy/llm.env` to the E4B file and run `sudo systemctl restart skyy-llm`. Flag changes go in `jetson/systemd/*.service`; re-run the script to install them.

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

### 10. Cloned voice (XTTS-v2, optional)

Piper has one voice per language, so Skyy sounds like a different person in English, Spanish and French. XTTS-v2 clones one voice from 30–60 s of recordings and speaks all three languages with it. It needs CUDA to keep up with speech, so it runs here, not on the laptop. There is no XTTS-v3: Coqui closed in 2024, and the code is kept up by Idiap as the `coqui-tts` package.

> **Rights first.** Only clone a voice you may use. A cartoon character's voice (e.g. Jabberjaw) belongs to the studio, and the recording is a real actor's voice: fine for a private test, but the public kiosk needs written permission, or a consenting person (a student or staff member) recording a similar character voice. **The XTTS-v2 weights are under the Coqui Public Model License: non-commercial use only**, and a commercial license can no longer be bought. Confirm with MDC that the kiosk counts as non-commercial.

1. Python environment with CUDA PyTorch. On JetPack 7 the standard ARM CUDA builds from PyTorch's own index work on the Orin. A plain `pip install torch` gives a CPU-only build, so always pass the index. The torch wheels and the NVIDIA libraries they pull in take several GB: with the SSD from step 2, put the venv, pip's cache and its temporary folder there first.
   ```bash
   export TMPDIR=/ssd/tmp PIP_CACHE_DIR=/ssd/pip-cache && mkdir -p $TMPDIR $PIP_CACHE_DIR   # SSD only
   python3 -m venv ~/skyy-tts-venv && source ~/skyy-tts-venv/bin/activate
   pip install -U pip
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130   # or .../whl/cu132
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability()); x = torch.randn(1024, 1024, device='cuda'); print((x @ x).sum().item())"
   pip freeze | grep -E '^torch' > ~/torch-pin.txt     # stops coqui-tts from swapping in a CPU-only torch
   export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.7   # DeepSpeed compiles its kernels for the Orin
   pip install "coqui-tts==0.26.2" "transformers==4.51.3" deepspeed ninja six fastapi uvicorn -c ~/torch-pin.txt
   python -c "import torch; print(torch.cuda.is_available())"     # still True
   ```
   The check must print `True`, `(8, 7)` and a number. A warning about compute capability 8.7 is harmless. If `cu130` fails, try `cu132`. On JetPack 6, PyTorch's index has no Jetson CUDA builds: use NVIDIA's wheel for your JetPack version (jetson-ai-lab) instead.

   **Why DeepSpeed and these old versions:** on the Orin, one CPU core handing work to the GPU limits plain XTTS to about real time (RTF ~1.05, first audio ~0.9 s, GPU only 35–64% busy). DeepSpeed's merged kernels brought it to **RTF ~0.55 and first audio ~0.48 s** (measured 2026-11-12, other servers stopped). DeepSpeed doesn't work with `transformers` 4.57+ (`'DynamicCache' object has no attribute 'dtype'`), and coqui-tts 0.28 needs 4.57+, so this pins coqui-tts 0.26.2, the newest release that accepts `transformers` 4.51. DeepSpeed compiles its kernels on the first start (several minutes); later starts reuse them. Without DeepSpeed, install `"coqui-tts[codec]"` and set `XTTS_DEEPSPEED=0`.
2. The model (~1.9 GB): `huggingface-cli download coqui/XTTS-v2 --local-dir ~/models/xtts_v2`
3. The voice: on the laptop, put the clips in `voices/<name>/` (for example `voices/jabberjaw/01.wav …`) and check them with `python 11_voice_ref.py voices/jabberjaw` (its docstring shows how to remove music with Demucs and cut clips with ffmpeg). Copy the folder to `~/voices/` on the Jetson. The clips aren't in git (`*.wav` is ignored).
4. Test by hand: `cd jetson && XTTS_DEEPSPEED=1 COQUI_TOS_AGREED=1 ~/skyy-tts-venv/bin/python tts_server.py`. It prints one `warm-up` line per language when it's ready (about 1 minute). Then from the laptop: `python 04_tts.py --engine xtts --jetson 192.168.55.1` (time to first audio per sentence, plays the three languages).
5. Run it as a service: `sudo ./jetson/install_services.sh --tts` (step 8). It writes `/etc/skyy/tts.env` with `XTTS_VENV=~/skyy-tts-venv` and `XTTS_DEEPSPEED=1`; if your venv is elsewhere, fix `XTTS_VENV=` there and run the script again. Logs: `journalctl -u skyy-tts -f`; each sentence logs its time to first audio and real-time factor (RTF below 1 = faster than speech).
6. On the laptop: `$env:SKYY_TTS = "xtts"` before `python 07_voice_webrtc.py` (or `--tts xtts` for step 6). `SKYY_TTS_VOICE` picks the voice folder (default `jabberjaw`). If the Jetson TTS can't be reached, Skyy keeps talking with Piper and tries XTTS again 30 s later.

Tuning in `/etc/skyy/tts.env`: `XTTS_TEMPERATURE` (default 0.65; lower is steadier, higher more expressive), `XTTS_SPEED` (1.0), `XTTS_CHUNK` (20; smaller starts sooner but can click between chunks), `XTTS_MAX_PER_CHAR` (0.1: a sentence stops after 0.1 s per character + 2 s of audio, logged as `cut off … (runaway)`; a short or noisy reference clip can make XTTS babble on), `XTTS_DEEPSPEED` and `XTTS_FP16` (0 or 1). XTTS shares the Orin's GPU with Gemma, which is still writing the answer while the first sentence is spoken: compare latencies with `SKYY_TTS=piper` and `xtts`.

---

## Part 2 – Laptop voice pipeline (Windows 11)

This runs on the Lenovo Aura. Whisper runs on the iGPU (`GPU`) by default. To use the Aura's NPU, pass `--stt-device NPU` (step 6) or set `$env:SKYY_STT_DEVICE="NPU"` (step 7). If a device fails to load, it falls back NPU → GPU → CPU. Step 3 benchmarks every available device (`--devices` picks some).

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
| `httpx` | Knowledge-base calls to EmbeddingGemma and Qdrant, and the LLM warm-up (`kb.py`) |
| `numpy`, `scipy`, `sounddevice`, `soundfile` | Audio capture, resampling and file I/O |
| `huggingface_hub` | Model downloads |

Models (downloaded to `models\`): **Whisper large-v3-turbo int8** (OpenVINO) and the Piper voices **en_US-lessac-medium**, **es_MX-claude-high** and **fr_FR-siwis-medium**.

### Scripts

Run them in order: each step checks one piece before the full loop.

| Script | What it does |
|---|---|
| `setup.ps1` | One-time setup: installs Python 3.13 with winget if needed, creates `.venv` (rebuilds it if it was made with another Python version), installs requirements, downloads models and runs `check_setup.py`. |
| `check_setup.py` | Checks Python version, packages, OpenVINO GPU, audio devices, models and Piper. `--jetson <ip>` also checks the LLM server. |
| `download_models.py` | Downloads Whisper and the Piper voices. `--small` adds Whisper small; `--voices <names>` adds more voices. Skips files already downloaded. |
| `01_mic.py` | Lists audio devices, records a few seconds, reports level and clipping, plays it back. `--save samples/en1.wav` records test clips. |
| `02_vad.py` | Silero VAD on the live mic or a file: prints when speech starts and ends, for tuning the silence wait (`--silence-ms`). |
| `03_whisper.py` | Benchmarks Whisper on CPU/GPU/NPU: latency, real-time factor, transcript and detected language. `--lang es` forces a language. |
| `04_tts.py` | Piper in English, Spanish and French: time to first audio per sentence, with playback. `--en/--es/--fr` compare voices. `--engine xtts --jetson 192.168.55.1` tests the cloned voice on the Jetson instead (`--save samples/xtts` keeps the audio). |
| `05_llm.py` | Gemma on the Jetson: time to first token and tokens/s for EN/ES/FR prompts. Warns if "thinking" is on. |
| `diag_stream.py` | Network diagnostic: repeats streaming requests over fresh connections to spot a flaky link. |
| `06_voice_loop.py` | **Full voice loop**, laptop mic and speakers (PyAudio). Half-duplex by default (mic ignored while Skyy talks, so it can't hear itself); `--barge-in` with headphones. |
| `07_voice_webrtc.py` | **Full voice loop through the browser** (http://localhost:7860, the Finn page in `web/`; Pipecat's own test page is at `/client/`). The browser does echo cancellation, noise suppression and auto gain, so it works on speakers and allows interrupting. Configured with `SKYY_*` environment variables. This is the version to build on. Answers from the MDC FAQ knowledge base (`SKYY_KB_MODE`, see `kb.py`). |
| `08_kb_index.py` | Loads the MDC FAQ (`mdc_faq_kb_multilingual.jsonl`: each FAQ in English, Spanish and French) into Qdrant: EmbeddingGemma vectors for every entry, then test searches with timings and scores (to pick `SKYY_KB_MIN_SCORE`). Run as `python 08_kb_index.py --host 192.168.55.1`; `--test` only searches. Needs `SKYY_QDRANT_KEY`. Only needed for `SKYY_KB_MODE=rag`: the default `prompt` mode doesn't use Qdrant. |
| `09_kb_latency.py` | Compares the knowledge-base modes (`off`, `prompt`, `rag`): time from question to the LLM's first token, retrieval included, and how many prompt tokens llama-server had to process. |
| `10_kb_eval.py` | Quality of the knowledge base, on the test set `kb_eval_questions.jsonl` (each FAQ asked in a visitor's own words in EN/ES/FR, misheard versions like "corner rule", and off-topic questions). Retrieval: hit@1, hit@3 and MRR per language, plus the recommended `SKYY_KB_MIN_SCORE`. Answers in `rag` and `prompt` mode: key facts present, right language, off-topic declined, and a Gemma grade (correct, grounded). Run as `python 10_kb_eval.py --host 192.168.55.1` (~25 min; `--retrieval-only` ~1 min). Needs the collection from step 08 and `SKYY_QDRANT_KEY`. Results go to `eval_results/`. `--judge gemini` grades with Gemini (default `gemini-3.5-flash`, `--gemini-model` to change) instead of Gemma; it needs `GEMINI_API_KEY` and sends the test questions, answers and FAQ to Google, so only use it on this written test set. `--judge-test eval_results\kb_eval_<...>.jsonl --judges gemma gemini-2.5-flash gemini-3.5-flash` checks the judges themselves: it plants known errors in copies of good answers (changed number, wrong office or system, key step removed, yes/no flipped, wrong topic, "I don't have that information", an invented sentence), plus good variants that must pass (shortened to the key sentences, same answer in another language), and reports how many each judge gets right and where they disagree. |
| `11_voice_ref.py` | Checks the reference clips for a cloned XTTS voice (`python 11_voice_ref.py voices/jabberjaw`): length, sample rate, mono, clipping, loudness and pauses per clip, and the total (aim for 30–60 s). See Part 1, step 10. |

Shared modules used by steps 6 and 7:

| Module | What it does |
|---|---|
| `ov_whisper_stt.py` | Pipecat speech-to-text service running Whisper on OpenVINO. Limits languages to EN/ES/FR: if Whisper detects another language (e.g. Italian or Portuguese from people nearby), it re-transcribes in the allowed languages and keeps the most confident result. Uses "Sky" and the FAQ's names (Gordon Rule, MDConnect, …) as a hint, so the wake word isn't heard as "Hi" (`SKYY_STT_HINT` changes the names). Drops repetition loops ("so, so, so, …") that Whisper produces on background noise. `SKYY_SAVE_AUDIO=1` saves each clip to `debug_audio/`. |
| `bilingual_tts.py` | Pipecat Piper service with one voice per language. Detects the language of each sentence and speaks it with the matching voice. `MultilingualXTTSService` (`SKYY_TTS=xtts`) sends each sentence with its language to the Jetson's XTTS server instead, so all three languages use one cloned voice; it holds a one-word opener ("Sí.") until the next clause, because XTTS garbles one-word inputs, and cuts inputs longer than XTTS's limit (~250 characters) at a comma. Both read links as their domain ("mdc.edu", not the whole URL) and drop markdown symbols. |
| `wake.py` | The "Sky" wake word. While asleep, speech without "Sky" is ignored. After "Sky", Skyy stays awake until 6 s after its last answer (`--wake-timeout` / `SKYY_WAKE_TIMEOUT`), so follow-up questions don't need the wake word. Other people's speech doesn't keep it awake, but a question that starts inside that window is never cut off: Skyy stays awake until its transcript arrives (at most 20 s), even if the visitor pauses mid-question. Includes common Whisper spellings (Skye, Skype, …). |
| `kb.py` | The MDC FAQ knowledge base. `SKYY_KB_MODE=prompt` (default): the whole English FAQ (~4k tokens) goes in the system prompt, which llama-server caches, so no per-turn retrieval. `rag`: `KnowledgeInjector` embeds each question with EmbeddingGemma, searches Qdrant and adds the top entries to the question (best language per FAQ, so the notes are usually in the visitor's language). `SKYY_KB` points to another FAQ file. `warm_up()` caches the system prompt at startup. |
| `skyy_processors.py` | `WakeWordJoiner`: if you say "Sky", pause, then ask, both parts are joined into one question instead of a greeting plus an interruption. `SideTalkFilter`: while Skyy is awake, speech without "Sky" that is under 3 words or an unfinished sentence ("Let me mute it for a") is ignored as side conversation; short replies like "Yes" or "Gracias" still count. `HistoryCap`: the LLM sees at most 6 messages; once over, the history is cut back to the last exchange, so old or misheard exchanges don't pile up and llama-server can reuse its cached prompt between cuts (`SKYY_HISTORY`, `SKYY_HISTORY_KEEP`). |
| `skyy_prompt.py` | Skyy's system prompt for step 7: language rule, short spoken answers, how to handle misheard words. `kb.py` adds the FAQ to it. |
| `web/` | The kiosk page step 7 serves at http://localhost:7860: Finn, the MDC mascot, in 3D. `finn3d.js` draws `finn.glb` with three.js (`vendor/`, MIT, no internet needed) in two states. **Idle:** mouth closed; the scene doesn't change, so nothing is redrawn. **Speaking:** the jaw opens and closes with the sound actually playing (`voice.js`: loudness opens it, the spectrum shapes it, so it stops if Skyy is interrupted); on `demo.wav` it follows the voice with 0.92 correlation and 10-20 ms lag. Around it, on damped springs so the motion has some weight: the snout lifts a little as the jaw drops, the head nods on stressed syllables (sudden rises in loudness) with a small random turn and tilt and drifts slowly otherwise, the body leans in and sways, and the eye blinks every few seconds. All of it eases in when Skyy starts and settles back to the rest pose within about a second after the voice ends, and then redrawing stops. He is also held idle from the moment the visitor starts talking until Skyy answers (at most 6 s after they stop), because Whisper runs on the same integrated GPU. Measured on the Aura (Whisper median per 5 s clip): page closed ~219 ms, a blank Edge window ~249 ms, idle Finn ~262 ms; speaking at the default 30 fps keeps the GPU ~9% busy (~279 ms; the jaw alone was ~5%, ~270 ms). `skyy.js` is a small WebRTC client in place of Pipecat's prebuilt one: keeps the browser's echo cancellation, shows what is happening (listening, Skyy speaking) and reconnects by itself if the bot restarts. `?demo` plays a recorded voice (`demo.wav`) without the bot (`python -m http.server 8000 --directory web`, then http://localhost:8000/?demo); `?open=0.5` holds the mouth half open, `?blink=1` shuts the eye, `?pitch=5`, `?yaw=5`, `?roll=5` hold the head (degrees), `?fps=60` changes the speaking frame rate. The older 2D Finn (`finn.js`, `finn.png`) is still at `/finn/2d.html`. |
| `make_finn_model.py` | Makes `web/finn.glb` from the mascot model (a Tripo `.glb`, not in the repo: one solid skin with no bones or jaw): cuts the skin along the lower lip; adds a skeleton (body, head, and jaw as a child of the head) with the head's weights blending smoothly through the neck; adds the inside of the mouth, a tongue (morph `grow`) and an eyelid over the painted eye (morph `blink`); and shrinks the 4K textures. In the rest pose it looks like the source model. Needs scipy (in the venv). `python make_finn_model.py "C:\path\to\model.glb"`. |

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
| 7 Browser loop | `$env:SKYY_JETSON="192.168.55.1"; python 07_voice_webrtc.py`, then open http://localhost:7860 in Chrome/Edge and click **Talk to Skyy** | Works on speakers without Skyy answering itself; Finn's mouth moves only while Skyy speaks |

Step 7 options (environment variables; the docstring at the top of `07_voice_webrtc.py` describes each one):

| Variable | Default | What it does |
|---|---|---|
| `SKYY_JETSON`, `SKYY_LLM_PORT` | `192.168.1.50`, `8080` | Where llama-server runs. Set `SKYY_JETSON=192.168.55.1` for USB-C. |
| `SKYY_HALF_DUPLEX` | off | `1` = mic ignored while Skyy talks (no interrupting) |
| `SKYY_NO_WAKE` | off | `1` = answer everything, no "Sky" needed |
| `SKYY_WAKE_WORDS`, `SKYY_WAKE_TIMEOUT` | Whisper spellings of "Sky", `6` | Wake word spellings; seconds awake after Skyy's last answer |
| `SKYY_STOP_SECS`, `SKYY_INCOMPLETE_SECS` | `0.4`, `1.5` | Silence that ends a turn; the same when the sentence sounds unfinished |
| `SKYY_HISTORY`, `SKYY_HISTORY_KEEP` | `6`, `1` | Most messages the LLM sees; exchanges kept when the history is cut |
| `SKYY_STT_DEVICE`, `SKYY_WHISPER`, `SKYY_STT_HINT` | `GPU`, large-v3-turbo int8, FAQ names | Whisper device, model folder and hint words |
| `SKYY_VOICE_EN/ES/FR` | the three default voices | Piper voice per language |
| `SKYY_KB_MODE` | `prompt` | Knowledge base: `prompt` (whole FAQ in the cached prompt), `rag` (Qdrant search, see step 08) or `off` |
| `SKYY_QDRANT_KEY`, `SKYY_KB_TOP_K`, `SKYY_KB_MIN_SCORE` | –, `3`, `0` | `rag` mode only: Qdrant key, entries per question, minimum score |
| `SKYY_SAVE_AUDIO` | off | `1` = save each Whisper clip to `debug_audio/` |

Don't pass `--host` to step 7; that sets the web server's own address.

Kiosk: start the browser full screen and allowed to play sound without a click, so the page connects on its own:
`msedge --kiosk http://localhost:7860 --edge-kiosk-type=fullscreen --autoplay-policy=no-user-gesture-required`
(Chrome: `chrome --kiosk --autoplay-policy=no-user-gesture-required http://localhost:7860`). Allow the microphone once for `localhost`.

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
- **pip fails building PyAudio or aiortc** → you're not on Python 3.13 (or 3.12) 64-bit; `python check_setup.py` shows the version.
- **"no GPU device"** → update the Intel driver and reboot; use `--devices CPU` / `--stt-device CPU` meanwhile.
- **Finn doesn't move, or no sound** → the browser blocked audio until a click: click **Talk to Skyy** (or use the kiosk flags above). If the status line stays on "Reconnecting…", step 7 isn't running or the mic was refused.
- **Want another Piper voice** → `python download_models.py --voices en_US-amy-medium`; browse voices at https://huggingface.co/rhasspy/piper-voices.
