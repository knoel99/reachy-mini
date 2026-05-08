#!/usr/bin/env bash
# Lance main.py avec les chemins natifs nécessaires :
#  - le plugin GStreamer Rust webrtcsrc compilé sous /opt/gst-plugins-rs
#  - les libstdc++/libgcc système (override miniconda dont les versions sont
#    trop anciennes pour gstlibav)
#  - les variables d'env du fichier .env

set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Fichier .env manquant. cp .env.example .env puis renseigner OPENAI_API_KEY et REACHY_HOST." >&2
  exit 1
fi

export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu:${GST_PLUGIN_PATH:-}
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
set -a; source .env; set +a

exec python -u main.py
