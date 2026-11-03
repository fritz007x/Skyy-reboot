#!/usr/bin/env bash
# Install the Skyy llama-servers as systemd services (restart on failure, start at boot).
#
#   sudo ./install_services.sh          # main LLM (:8080) + EmbeddingGemma (:8082)
#   sudo ./install_services.sh --laya   # also Laya (:8081)
#
# Model paths go in /etc/skyy/*.env. Existing .env files are kept, so re-running is safe.
set -euo pipefail

if [[ $EUID -ne 0 || -z "${SUDO_USER:-}" ]]; then
  echo "Run with sudo from your normal user: sudo $0 $*" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="$SUDO_USER"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

services=(skyy-llm skyy-embed)
[[ "${1:-}" == "--laya" ]] && services+=(skyy-laya)

declare -A default_model=(
  [llm]="$RUN_HOME/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
  [embed]="$RUN_HOME/models/embeddinggemma-2-UD-Q4_K_XL.gguf"
  [laya]="$RUN_HOME/models/<laya-file>.gguf"
)

mkdir -p /etc/skyy
for svc in "${services[@]}"; do
  name="${svc#skyy-}"
  env="/etc/skyy/$name.env"
  [[ -f "$env" ]] || echo "MODEL=${default_model[$name]}" > "$env"
  model="$(sed -n 's/^MODEL=//p' "$env")"
  [[ -f "$model" ]] || echo "WARNING: $svc: model not found: $model (fix MODEL= in $env)" >&2

  sed -e "s|@USER@|$RUN_USER|g" -e "s|@HOME@|$RUN_HOME|g" \
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
