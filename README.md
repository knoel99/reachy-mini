# reachy-mini

Reachy Mini contrôlé par un LLM via deux pipelines vocaux au choix : **OpenAI
Realtime** (audio bidirectionnel) ou **Grok chat-with-tools** (mic → STT → chat
avec function-calling). Dans les deux cas le robot n'a pas de voix : il ne
réagit qu'à travers des appels d'outils — mouvements de tête, antennes,
chorégraphies, sons d'émotion préenregistrés.

## Structure

```
reachy-mini/
├── pyproject.toml
├── run.sh                        # charge .env, GST_PLUGIN_PATH, LD_PRELOAD
├── docs/DEPLOY_ON_ROBOT.md       # déploiement systemd sur le Pi
├── scripts/export_emotion_sounds.py
├── colab/fastvlm_7b_server.ipynb # serveur FastVLM-7B (Colab GPU)
├── src/reachy_voice/             # pipeline voix
│   ├── __main__.py
│   ├── _actions.py               # RobotActions : dispatch tool → robot
│   ├── _log.py
│   ├── emotions.py               # EmotionPlayer (preload + push_audio_sample)
│   ├── tools.py                  # build_tools, build_instructions, LOOK_POSES
│   ├── openai/realtime.py        # bridge WebSocket
│   └── grok/{vad,stt,chat}.py    # mic → VAD → STT → chat-with-tools
└── src/reachy_vision/            # package vision dédié (optionnel)
    ├── camera.py                 # Camera : dernière frame BGR sous lock
    ├── moondream.py              # Moondream : caption() + point()
    ├── fastvlm.py                # FastVLM : caption() (HTTP vers Colab)
    └── preview.py                # Preview : MJPEG :5050 + overlays debug
```

Installation : `pip install -e .` (voir [INSTALL.md](./INSTALL.md)).
Pour la vision : `pip install -e '.[vision]'` (ajoute Moondream).
Lancement : `./run.sh <model>` ou `python -m reachy_voice` ou `reachy-voice`.

### Activer la vision (optionnel)

Copie `.env.example` vers `.env` puis configure un backend :

```bash
# Option A — Moondream cloud (recommandé pour démarrer)
VISION_BACKEND=moondream
MOONDREAM_API_KEY=...    # https://moondream.ai/c/account/api-keys

# Option B — FastVLM-7B servi par Colab (voir colab/fastvlm_7b_server.ipynb)
VISION_BACKEND=fastvlm
FASTVLM_URL=https://xxxx.trycloudflare.com
```

Deux outils sont alors exposés au LLM :
- `look_and_describe(question)` — capture une frame et pose une question
- `find_object(target)` — localise un objet et oriente la tête
  (Moondream uniquement — utilise `point()` natif)

Une preview web de debug tourne sur `http://localhost:5050` dès que le
backend vision démarre. Elle stream le MJPEG en direct et overlaye la
dernière caption + le crosshair du dernier `find_object` (chaque
overlay fade après 5 s). Désactiver avec `VISION_PREVIEW=0`. La
preview bind sur `0.0.0.0` — firewall ton LAN ou désactive-la en
prod sur le Pi.

## Les deux pipelines

### OpenAI Realtime (`provider=openai`)

```
mic → PCM 24 kHz → WebSocket → server VAD → STT → modèle (tool calls
en streaming) → response.done → dispatch sur le robot
```

`output_modalities=["text"]` désactive le TTS. Le modèle peut faire du
reasoning interleaved (`gpt-realtime`, `gpt-realtime-2`).

### Grok chat-with-tools (`provider=grok`)

```
mic → PCM 16 kHz → webrtcvad local → utterance → POST /v1/stt →
text → POST /v1/chat/completions (tools) → tool_calls → dispatch →
re-call chat avec résultats → loop
```

Pas d'API realtime. ~100× moins cher par tour pour un usage tool-only,
latence plus prévisible (pas de TTS gaspillé). Contexte natif du modèle,
aucune troncature manuelle.

## Démarrage

1. **Installation** (Python + GStreamer + plugin Rust webrtcsrc) : voir
   **[INSTALL.md](./INSTALL.md)**.
2. **Configurer `.env`** :
   ```bash
   cp .env.example .env
   # éditer : OPENAI_API_KEY ou XAI_API_KEY, REACHY_HOST=<ip-LAN-du-robot>
   ```
3. **Lancer** — le shortcut est le nom complet du modèle, le provider
   est déduit du préfixe :
   ```bash
   # OpenAI Realtime
   ./run.sh gpt-realtime-mini
   ./run.sh gpt-realtime
   ./run.sh gpt-realtime-2

   # Grok chat-with-tools
   ./run.sh grok-4-1-fast-non-reasoning
   ./run.sh grok-4-1-fast-reasoning

   # Override explicite si besoin
   ./run.sh openai <model>
   ./run.sh grok   <model>

   # Sans argument : valeurs par défaut depuis .env
   ./run.sh
   ```

## Choix du modèle

### OpenAI Realtime

| Modèle              | Audio in/out / 1M | reasoning.effort | Note                                             |
| ------------------- | ----------------- | ---------------- | ------------------------------------------------ |
| `gpt-realtime-mini` | $10 / $20         | non              | Faible coût, conversation simple                 |
| `gpt-realtime`      | $32 / $64         | oui              | Tool calling robuste                             |
| `gpt-realtime-2`    | $32 / $64         | oui              | **Latest**, plans multi-étapes (chorégraphies)   |

`reasoning.effort` ∈ `minimal | low | medium | high`, défaut `medium`,
configurable via `OPENAI_REASONING_EFFORT`.

### Grok chat

| Modèle                          | Input / 1M | Output / 1M | Note                              |
| ------------------------------- | ---------- | ----------- | --------------------------------- |
| `grok-4-1-fast-non-reasoning`   | $0.20      | $0.50       | Le plus rapide, défaut            |
| `grok-4-1-fast-reasoning`       | $0.20      | $0.50       | Planning sur séquences complexes  |

À cela s'ajoute le STT REST : **$0.10/hr** d'audio transcrit.

## Logs

Chaque event-line est préfixé `[HH:MM:SS.mmm +Δs]` (delta depuis le
log précédent — pratique pour mesurer la latence par étape).

Au démarrage : `[config] provider=… model=… prices …`. À chaque tour :
`cost=$X cumul=$Y` + breakdown des tokens. À la fin : `[cost] session
total: $Z over N turn(s)`.

## Tools exposés au modèle

Le modèle invoque ces fonctions via function-calling (jamais en texte) :

- **`play_emotion(name)`** — émotion préenregistrée (mouvement + son
  audio joint). Enum populé dynamiquement depuis le dataset HF
  `pollen-robotics/reachy-mini-emotions-library` (~80 émotions).
- **`look(direction)`** — tête vers `left`, `right`, `up`, `down`,
  `center`.
- **`move_sequence(steps)`** — chorégraphie planifiée. Chaque step :
  `yaw`, `pitch`, `roll` (deg), optionnellement `antenna_left`/`antenna_right`
  (deg) et `duration` (s). Pour les cercles, hochements, danses,
  imitations.

Les trois outils sont implémentés dans
[`_actions.py`](./src/reachy_voice/_actions.py) et partagés entre les deux
providers.

## Trouver l'IP du robot

- **Linux natif** : `ping -4 reachy-mini.local`
- **WSL2** : depuis PowerShell Windows, `ping -4 reachy-mini.local`
  (le mDNS WSL2 ne résout pas les `.local` par défaut)

## Voir aussi

- [INSTALL.md](./INSTALL.md) — installation + dépannage
- [docs/DEPLOY_ON_ROBOT.md](./docs/DEPLOY_ON_ROBOT.md) — déploiement systemd sur le Pi
- [OpenAI Realtime API](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [xAI Voice Agent API](https://docs.x.ai/developers/model-capabilities/audio/voice-agent)
- [xAI Speech-to-Text REST](https://docs.x.ai/developers/rest-api-reference/inference/voice)
- [xAI Models & Pricing](https://docs.x.ai/developers/models)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
