# Installation

Connexion d'un Reachy Mini Wireless à OpenAI Realtime (`gpt-realtime-mini`)
depuis une machine Linux/WSL2 sur le même LAN que le robot.

> ⚠️ Sur Linux, le plugin GStreamer Rust `webrtcsrc` n'est pas packagé en
> apt et doit être compilé. Compter ~15-20 min la première fois (Rust +
> compilation native).

---

## 1. Prérequis système

- Reachy Mini Wireless allumé, daemon démarré, joignable sur le LAN.
- Linux (testé Ubuntu 24.04, WSL2 supporté).
- Python 3.10+ (3.13 OK).
- Une clé API OpenAI avec accès à `gpt-realtime-mini`.

## 2. Dépendances système (apt)

```bash
sudo apt-get update
sudo apt-get install -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libssl-dev \
    libnice10 \
    gstreamer1.0-nice \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
    libportaudio2 \
    python3-gi \
    python3-gi-cairo \
    pkg-config \
    python3-dev \
    python3-venv
```

> 💡 **`python3-venv` indispensable** : sur Ubuntu/Debian, sans ce paquet
> `python3 -m venv .venv` ne crée que les symlinks Python — **pas** le
> script `activate` ni `pip` (le module `ensurepip` est absent).

> 💡 **Pourquoi `libgirepository1.0-dev` et pas `-2.0-dev`** : `reachy-mini`
> dépend de PyGObject ≤ 3.46.0 qui requiert l'API v1 de
> gobject-introspection.

## 3. Plugin GStreamer Rust `webrtcsrc`

### 3.1. Rust + cargo-c

```bash
# Rust (si absent)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# cargo-c (build tool pour les plugins C-ABI Rust)
# Note : cargo-c 0.10.22+ exige rustc 1.93+. Avec rustc 1.91 utiliser 0.10.20.
cargo install cargo-c --version 0.10.20 --locked
```

### 3.2. Compiler `gst-plugin-webrtc`

```bash
git clone --depth 1 --branch 0.14.1 \
    https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git /tmp/gst-plugins-rs

sudo mkdir -p /opt/gst-plugins-rs && sudo chown $USER /opt/gst-plugins-rs

cd /tmp/gst-plugins-rs
cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release
```

⏱ Compter 10-15 min. À la fin, on doit avoir
`/opt/gst-plugins-rs/lib/x86_64-linux-gnu/libgstrswebrtc.so`.

### 3.3. Exporter le chemin

```bash
echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu:$GST_PLUGIN_PATH' \
    >> ~/.bashrc
source ~/.bashrc
```

Vérifier :

```bash
gst-inspect-1.0 webrtcsrc | head
# doit afficher la fiche du plugin, pas "No such element..."
```

> 💡 ARM64 : remplacer `x86_64-linux-gnu` par `aarch64-linux-gnu`.

## 4. Dépendances Python

```bash
cd ~/github/reachy-mini

# Idéalement un venv neuf (sans héritage conda)
deactivate 2>/dev/null; conda deactivate 2>/dev/null
unset PYTHONPATH
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> Le projet expose un package `reachy_voice` via `pyproject.toml`. Le
> `-e` installe en mode éditable pour pouvoir modifier le code sans
> réinstaller. Ça crée aussi le script `reachy-voice` dans le venv.

## 5. Configuration

```bash
cp .env.example .env
$EDITOR .env
```

Renseigner au minimum :

- `OPENAI_API_KEY` — votre clé OpenAI
- `REACHY_HOST` — IP **LAN** du Reachy Mini

### Trouver l'IP du robot

- **Réseau natif Linux** : `ping -4 reachy-mini.local`
- **WSL2** (mDNS ne marche pas par défaut) : depuis PowerShell Windows
  `ping -4 reachy-mini.local` puis recopier l'IP `192.168.x.x`.

### Vérifier la connectivité

```bash
ping -c 2 $REACHY_HOST
curl -sS http://$REACHY_HOST:8000/docs | head -c 100   # doit renvoyer du HTML
nc -zv $REACHY_HOST 8443                               # doit dire "succeeded"
```

> 💡 **WSL2 + LAN** : si `ping` du robot échoue, activer le mode
> `mirrored` de WSL2. Créer/éditer `C:\Users\<user>\.wslconfig` :
> ```
> [wsl2]
> networkingMode=mirrored
> ```
> puis `wsl --shutdown` dans PowerShell et redémarrer WSL.

## 6. Lancement

Le plus simple : utiliser le wrapper `run.sh` qui pose `GST_PLUGIN_PATH`,
`LD_PRELOAD` (libstdc++/libgcc système — important si vous tournez
sous miniconda dont les versions sont trop vieilles), et charge `.env` :

```bash
./run.sh
```

Au démarrage on doit voir :

```
INFO:reachy_mini.media.webrtc_client_gstreamer:Audio send chain ready (bidirectional audio enabled)
[audio] mic=16000 Hz, speaker=16000 Hz
[ws] connected
```

Puis parler au robot — la VAD côté serveur OpenAI segmente votre tour
de parole, le robot répond en audio et joue les émotions.

`Ctrl+C` pour arrêter.

### Sans wrapper (équivalent manuel)

```bash
export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu:$GST_PLUGIN_PATH
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1"
set -a; source .env; set +a
python -m reachy_voice
```

---

## Dépannage

### `Failed to connect to ws://reachy-mini.local:8000/ws/sdk: [Errno -2] Name or service not known`

mDNS `.local` non résolu (typique WSL2). Renseigner `REACHY_HOST=<ip>` dans `.env`.

### WebRTC `TimeoutError: timed out` sur `:8443`

Le daemon rapporte une `wlan_ip` non routable depuis votre machine
(autre interface du robot). Le Le bridge patche ça en forçant
`wlan_ip = REACHY_HOST` quand cette variable est définie.

### `RuntimeError: Failed to create webrtcsrc element. Is the GStreamer webrtc rust plugin installed?`

Voir étape 3. Vérifier ensuite :
```bash
echo $GST_PLUGIN_PATH
gst-inspect-1.0 webrtcsrc
```

### `Failed to load plugin libgstlibav.so: GLIBCXX_3.4.32 / GCC_12.0.0 not found`

Vous tournez sous miniconda dont `libstdc++`/`libgcc_s` sont plus
vieux que ceux requis par `libgstlibav` (paquet apt). `run.sh` règle
ça via `LD_PRELOAD` du libstdc++/libgcc système. En lancement manuel :

```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1"
```

### `Unknown parameter: 'session.modalities'` ou `Missing required parameter: 'session.audio.output.format.rate'`

L'API GA Realtime a remplacé `modalities` par `output_modalities` et
mis l'audio sous `session.audio.{input,output}.format.{type,rate}`.
Le bridge est déjà à jour ; cette erreur signifie que vous avez une
version pré-GA du code ou un fork divergent.

### `wake_up()` `TimeoutError: Task did not complete in time.`

Le daemon rapporte `backend_status.ready=False` même quand les moteurs
tournent. Le bridge capture l'exception et continue sans wake_up.
La conversation marche, mais l'animation de "réveil" est sautée.

### `error: cannot install package 'cargo-c …', it requires rustc 1.93 or newer`

Votre Rust est < 1.93. Soit mettre Rust à jour (`rustup update stable`),
soit utiliser `cargo install cargo-c --version 0.10.20 --locked`.

### `PyGObject build failed: Package 'gobject-introspection-1.0' was not found`

Installer `libgirepository1.0-dev` (et **pas** seulement `-2.0-dev`).

### `.venv/bin/activate` n'existe pas après `python3 -m venv`

Le `bin/` ne contient que les symlinks `python`, `python3`, `python3.12`,
sans `activate` ni `pip`. Cause : le paquet `python3-venv` (ou
`python3.X-venv` pour votre version) n'est pas installé, donc `ensurepip`
échoue silencieusement et la création du venv s'arrête à mi-chemin.

```bash
sudo apt install python3-venv
rm -rf .venv
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
```

### Le venv hérite des paquets miniconda

Si `pip install` affiche `Requirement already satisfied: ... in
/home/<user>/miniconda3/...`, votre venv n'est pas isolé. Refaire :

```bash
deactivate; conda deactivate
unset PYTHONPATH
rm -rf .venv
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
```

---

## Architecture

- Le bridge ouvre un WebSocket vers `wss://api.openai.com/v1/realtime?model=gpt-realtime-mini`
- Audio mic Reachy → resampling 24 kHz PCM16 → `input_audio_buffer.append`
- `response.output_audio.delta` (ou `response.audio.delta` pré-GA) →
  resampling vers la sortie Reachy → `media.push_audio_sample`
- VAD : `server_vad` (threshold 0.6, silence 700 ms)
- Transcription : `gpt-4o-mini-transcribe`, `language: "fr"`
- Voix de sortie : `alloy` par défaut (configurable via
  `OPENAI_REALTIME_VOICE`). Voix GA disponibles : `alloy`, `ash`,
  `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`.
- **Émotions hybrides** sur le robot, depuis le dataset HF
  `pollen-robotics/reachy-mini-emotions-library` (81 émotions) :
  - Auto sur events WS : `speech_started` → `attentive1`,
    `speech_stopped` → `thoughtful1`, `response.done` → `serenity1`
  - Sur tool call `play_emotion(name)` : le modèle appelle l'outil
    quand il veut exprimer `happy`, `surprised1`, `curious1`, etc.

## Schéma `session.update` (API GA)

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "output_modalities": ["audio"],
    "instructions": "...",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "gpt-4o-mini-transcribe", "language": "fr"},
        "turn_detection": {"type": "server_vad", "threshold": 0.6,
                           "prefix_padding_ms": 300, "silence_duration_ms": 700}
      },
      "output": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "alloy"
      }
    },
    "tools": [...],
    "tool_choice": "auto"
  }
}
```

## Références

- [OpenAI Realtime API – WebSocket](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [OpenAI gpt-realtime-mini](https://developers.openai.com/api/docs/models/gpt-realtime-mini)
- [Reachy Mini SDK (GitHub)](https://github.com/pollen-robotics/reachy_mini)
- [Reachy Mini GStreamer install (HF)](https://huggingface.co/docs/reachy_mini/en/SDK/gstreamer-installation)
- [`reachy-mini-emotions-library` (HF dataset)](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
