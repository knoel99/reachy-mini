# Voice API Bridges

Ce package contient les implémentations des bridges pour différents providers d'API Voice.

## Structure

- `voice_bridge.py` - Classe abstraite `VoiceBridge` définissant l'interface commune
- `openai_bridge.py` - Implémentation pour l'API OpenAI Realtime
- `grok_bridge.py` - Implémentation pour l'API xAI Grok Voice Think Fast

## Utilisation

```python
from bridges import VoiceBridge, OpenAIRealtimeBridge, GrokVoiceBridge

# Créer un bridge OpenAI
bridge = OpenAIRealtimeBridge(mini, model="gpt-realtime-2", voice="alloy")
bridge.run()

# Créer un bridge Grok
bridge = GrokVoiceBridge(mini, model="grok-voice-think-fast-1.0", voice="eve")
bridge.run()
```

## Méthodes abstraites à implémenter

Les sous-classes doivent implémenter:
- `get_ws_url()` - URL WebSocket
- `get_auth_header()` - Header d'authentification
- `get_session_config()` - Configuration de session
- `handle_audio_delta()` - Traitement audio
- `handle_transcript_delta()` - Traitement transcription
- `handle_input_transcription()` - Transcription input
- `supports_truncate()` - Support de conversation.item.truncate
- `compute_cost()` - Calcul des coûts
- `get_tools()` - Configuration des outils
- `_print_config()` - Affichage configuration

## Fonctionnalités communes

Gérées dans la classe de base `VoiceBridge`:
- Gestion WebSocket
- Stream audio microphone
- Barge-in (avec ou sans troncation)
- Gestion des outils (play_emotion, look, move_sequence)
- Exécution mouvements robot
- Playback émotions
