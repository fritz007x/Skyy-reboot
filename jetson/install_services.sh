#!/usr/bin/env bash
# Install the Skyy servers as systemd services (restart on failure, start at boot).
#
#   sudo ./install_services.sh               # main LLM (:8080) + EmbeddingGemma (:8082)
#   sudo ./install_services.sh --laya        # also Laya (:8081)
#   sudo ./install_services.sh --tts         # also the XTTS-v2 cloned voice (:8083, tts_server.py)
#   sudo ./install_services.sh --laya --tts  # both
#
# Model paths go in /etc/skyy/*.env. Existing .env files are kept, so re-running is safe.
# skyy-tts runs tts_server.py from this folder, so keep the folder where it is.
set -euo pipefail

if [[ $EUID -ne 0 || -z "${SUDO_USER:-}" ]]; then
  echo "Run with sudo from your normal user: sudo $0 $*" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="$SUDO_USER"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

services=(skyy-llm skyy-embed)
for arg in "$@"; do
  case "$arg" in
    --laya) services+=(skyy-laya) ;;
    --tts)  services+=(skyy-tts) ;;
    *) echo "Unknown option: $arg (use --laya and/or --tts)" >&2; exit 1 ;;
  esac
done

declare -A default_model=(
  [llm]="$RUN_HOME/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
  [embed]="$RUN_HOME/models/embeddinggemma-2-UD-Q4_K_XL.gguf"
  [laya]="$RUN_HOME/models/<laya-file>.gguf"
)

mkdir -p /etc/skyy
for svc in "${services[@]}"; do
  name="${svc#skyy-}"
  env="/etc/skyy/$name.env"
  if [[ "$name" == tts ]]; then
    [[ -f "$env" ]] || printf 'XTTS_MODEL_DIR=%s\nXTTS_VOICES_DIR=%s\n' \
      "$RUN_HOME/models/xtts_v2" "$RUN_HOME/voices" > "$env"
    model="$(sed -n 's/^XTTS_MODEL_DIR=//p' "$env")"
    voices="$(sed -n 's/^XTTS_VOICES_DIR=//p' "$env")"
    [[ -f "$model/model.pth" ]] || echo "WARNING: $svc: no XTTS model in $model (fix XTTS_MODEL_DIR= in $env)" >&2
    ls "$voices"/*/*.wav >/dev/null 2>&1 || echo "WARNING: $svc: no voice clips in $voices/<voice>/*.wav" >&2
    [[ -x "$RUN_HOME/skyy-tts-venv/bin/python" ]] || echo "WARNING: $svc: $RUN_HOME/skyy-tts-venv not found (see README)" >&2
  else
    [[ -f "$env" ]] || echo "MODEL=${default_model[$name]}" > "$env"
    model="$(sed -n 's/^MODEL=//p' "$env")"
    [[ -f "$model" ]] || echo "WARNING: $svc: model not found: $model (fix MODEL= in $env)" >&2
  fi

  sed -e "s|@USER@|$RUN_USER|g" -e "s|@HOME@|$RUN_HOME|g" -e "s|@HERE@|$HERE|g" \
    "$HERE/systemd/$svc.service" > "/etc/systemd/system/$svc.service"
done

# A llama-server started by hand in a terminal holds the port and makes the service fail
if pgrep -x llama-server >/dev/null; then
  echo "WARNING: llama-server is already running; stop it first or the services can't bind their ports:" >&2
  pgrep -ax llama-server >&2
fi

systemctl daemon-reload
systemctl enable --now "${services[@]}"
systemctl --no-pager status "${services[@]}" | grep -E '●|Active:'
echo
echo "Logs: journalctl -u skyy-llm -f   (models take a while to load; check with curl localhost:8080/v1/models)"
[[ " ${services[*]} " == *" skyy-tts "* ]] && echo "      journalctl -u skyy-tts -f   (ready when the warm-up lines appear; curl localhost:8083/health)"
exit 0
