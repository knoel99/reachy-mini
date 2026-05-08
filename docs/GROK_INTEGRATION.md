# Intégration de l'API Grok Voice Think Fast

Le projet supporte deux providers d'API Voice via une architecture
modulaire : OpenAI Realtime (par défaut) et xAI Grok Voice Think Fast.

> Les deux providers sont configurés en mode **text-only** : le modèle
> ne génère pas de TTS, le robot réagit uniquement via des appels
> d'outils (mouvements + sons d'émotion préenregistrés).

## Structure du code

```
src/reachy_voice/
├── __main__.py              # entrypoint (python -m reachy_voice)
├── tools.py                 # INSTRUCTIONS, LOOK_POSES, build_tools
├── emotions.py              # EmotionPlayer (preload + push_audio_sample)
└── bridges/
    ├── base.py              # VoiceBridge abstraite (transport, mic, tools)
    ├── openai.py            # OpenAIRealtimeBridge
    └── grok.py              # GrokVoiceBridge
```

## Différences entre les API

| Caractéristique           | OpenAI Realtime                                  | Grok Voice Think Fast                  |
| ------------------------- | ------------------------------------------------ | -------------------------------------- |
| Endpoint                  | `wss://api.openai.com/v1/realtime`               | `wss://api.x.ai/v1/realtime`           |
| Auth                      | `OPENAI_API_KEY`                                 | `XAI_API_KEY`                          |
| Modèles                   | gpt-realtime-mini, gpt-realtime, gpt-realtime-2  | grok-voice-think-fast-1.0              |
| `reasoning.effort`        | ✅ Supporté (gpt-realtime, gpt-realtime-2)       | ❌ Non supporté                        |
| Transcription input       | gpt-4o-mini-transcribe (forcée FR)               | Auto-détection                         |
| `output_modalities=text`  | ✅ Honoré côté API                               | ⚠️ Non documenté — audio droppé local  |

## Configuration (.env)

```bash
# Provider: openai (défaut) ou xai
VOICE_PROVIDER=openai

# Clés API
OPENAI_API_KEY=sk-...
XAI_API_KEY=...

# Modèles (optionnel — defaults explicites dans __main__.py)
OPENAI_REALTIME_MODEL=gpt-realtime-mini
GROK_MODEL=grok-voice-think-fast-1.0
```

## Lancement

```bash
# OpenAI (par défaut)
./run.sh              # via .env
./run.sh mini         # gpt-realtime-mini
./run.sh full         # gpt-realtime
./run.sh full2        # gpt-realtime-2

# xAI Grok
./run.sh grok         # grok-voice-think-fast-1.0

# Direct (équivalent)
python -m reachy_voice --provider openai --model gpt-realtime-2
python -m reachy_voice --provider xai    --model grok-voice-think-fast-1.0
```

## Méthodes à implémenter dans une sous-classe `VoiceBridge`

- `get_ws_url()` — URL WebSocket du provider
- `get_auth_header()` — Header d'authentification
- `get_session_config()` — Configuration de session (text-only,
  pas de bloc `audio.output`, pas de `voice`)
- `handle_transcript_delta()` — Affichage du texte streamé (debug)
- `handle_input_transcription()` — Affichage de la transcription
  utilisateur
- `compute_cost()` — Calcul des coûts à partir du bloc `usage`
- `_print_config()` — Bannière de configuration au démarrage

## Fonctionnalités mutualisées (`bridges/base.py`)

- WebSocket : connexion, queue d'envoi, parsing des événements
- Stream audio microphone (mic → resampling → PCM16 → base64 → send)
- Barge-in : `response.cancel` quand l'utilisateur reprend la parole
- Tool dispatch : `play_emotion`, `look`, `move_sequence`
- Exécution des mouvements (off-thread WS pour ne pas bloquer la pump)
- Tracking de coût per-tour et cumulé

## Prix Grok (placeholders)

Les prix dans `bridges/grok.py` (dict `PRICING`) sont des estimations à
remplacer par les valeurs officielles xAI :

- Text input  : $4.00 / 1M tokens
- Text output : $24.00 / 1M tokens
- Audio input : $32.00 / 1M tokens
- Audio output : 0 (mode text-only, le bloc `audio.output` n'est pas
  envoyé dans `session.update`)

## Dépannage

**`XAI_API_KEY is not set`** — Vérifier `.env` si `VOICE_PROVIDER=xai`.

**Le robot répond en anglais** — Grok détecte automatiquement la langue.
Ajouter une consigne FR explicite dans `INSTRUCTIONS` (`tools.py`).
