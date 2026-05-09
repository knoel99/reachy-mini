#!/usr/bin/env bash
# Lance le bridge avec les chemins natifs nécessaires :
#  - le plugin GStreamer Rust webrtcsrc compilé sous /opt/gst-plugins-rs
#  - les libstdc++/libgcc système (override miniconda dont les versions sont
#    trop anciennes pour gstlibav)
#  - les variables d'env du fichier .env
#
# Le package `reachy_voice` doit être installé dans le venv :
#     .venv/bin/pip install -e .

set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Fichier .env manquant. cp .env.example .env puis renseigner les clés API et REACHY_HOST." >&2
  exit 1
fi

export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu:${GST_PLUGIN_PATH:-}
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
set -a; source .env; set +a

# Shorthand: `./run.sh <model>` selects the provider automatically from
# the model name. Any extra args are forwarded to the bridge.
#
# OpenAI Realtime (audio-in, text-out):
#   ./run.sh gpt-realtime-mini
#   ./run.sh gpt-realtime
#   ./run.sh gpt-realtime-2
#
# Grok chat-with-tools (mic → STT → chat → tools):
#   ./run.sh grok-4-1-fast-non-reasoning
#   ./run.sh grok-4-1-fast-reasoning
#
# Explicit provider override:
#   ./run.sh openai <model>
#   ./run.sh grok   <model>
EXTRA=()
case "${1:-}" in
  gpt-realtime-mini|gpt-realtime|gpt-realtime-2)
    EXTRA+=(--provider openai --model "$1"); shift ;;
  grok-4-1-fast-non-reasoning|grok-4-1-fast-reasoning)
    EXTRA+=(--provider grok   --model "$1"); shift ;;
  openai|grok)
    EXTRA+=(--provider "$1")
    shift
    if [ -n "${1:-}" ]; then
      EXTRA+=(--model "$1")
      shift
    fi
    ;;
esac

exec python -u -m reachy_voice "${EXTRA[@]}" "$@"
