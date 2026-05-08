# Voice API bridges

Implémentations modulaires pour différents providers d'API Voice
Realtime, toutes en mode **text-only** (le modèle n'émet pas de TTS,
le robot réagit uniquement via tool calls).

## Fichiers

- `base.py` — Classe abstraite `VoiceBridge` : transport WebSocket,
  pump mic, dispatch des tool calls, barge-in.
- `openai.py` — `OpenAIRealtimeBridge`. Utilise
  `output_modalities: ["text"]`.
- `grok.py` — `GrokVoiceBridge`. Configuration équivalente, sans bloc
  `audio.output` et sans paramètre `voice`.

## Usage

```python
from reachy_mini import ReachyMini
from reachy_voice.bridges import OpenAIRealtimeBridge, GrokVoiceBridge

with ReachyMini() as mini:
    bridge = OpenAIRealtimeBridge(mini, model="gpt-realtime-mini")
    # ou : GrokVoiceBridge(mini, model="grok-voice-think-fast-1.0")
    bridge.run()
```

Les outils (`play_emotion`, `look`, `move_sequence`) et le prompt
système (`INSTRUCTIONS`) sont définis dans `reachy_voice/tools.py` ;
la lecture des sons d'émotion vit dans `reachy_voice/emotions.py`.

## Méthodes à implémenter dans une sous-classe

- `get_ws_url()` — URL WebSocket du provider
- `get_auth_header()` — Header `Authorization`
- `get_session_config()` — Payload de `session.update`
- `handle_transcript_delta()` — Affichage debug du texte streamé
- `handle_input_transcription()` — Transcription utilisateur
- `compute_cost()` — Coût par tour à partir du bloc `usage`
- `_print_config()` — Bannière de démarrage

## Fonctionnalités mutualisées

- WebSocket : connexion, queue d'envoi, parsing des événements
- Stream mic : capture → resample → PCM16 → base64 → send
- Barge-in : `response.cancel` quand l'utilisateur reprend la parole
- Tool dispatch : `play_emotion`, `look`, `move_sequence`
- Tracking coût par tour et cumulé
