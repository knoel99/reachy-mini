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

# Shorthand: `./run.sh <provider> <model>` overrides provider/model without
# editing .env. Any extra args are forwarded to the bridge.
#
# OpenAI provider (default):
#   ./run.sh mini    → provider=openai model=gpt-realtime-mini
#   ./run.sh full    → provider=openai model=gpt-realtime
#   ./run.sh full2   → provider=openai model=gpt-realtime-2
#
# xAI/Grok provider:
#   ./run.sh grok    → provider=xai model=grok-voice-think-fast-1.0
#
# Custom:
#   ./run.sh openai custom-model
#   ./run.sh xai custom-model
EXTRA=()
case "${1:-}" in
  mini)  EXTRA+=(--provider openai --model gpt-realtime-mini); shift ;;
  full)  EXTRA+=(--provider openai --model gpt-realtime);      shift ;;
  full2) EXTRA+=(--provider openai --model gpt-realtime-2);    shift ;;
  grok)  EXTRA+=(--provider xai --model grok-voice-think-fast-1.0); export GROK_DEBUG_USAGE=1; shift ;;
  openai|xai)
    EXTRA+=(--provider "$1")
    shift
    if [ -n "${1:-}" ]; then
      EXTRA+=(--model "$1")
      shift
    fi
    ;;
esac

exec python -u -m reachy_voice "${EXTRA[@]}" "$@"
