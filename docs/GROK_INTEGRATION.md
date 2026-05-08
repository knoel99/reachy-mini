# Intégration de l'API Grok Voice Think Fast

Ce document décrit l'intégration de l'API Grok Voice Think Fast (xAI) dans le projet reachy-mini.

## Architecture modulaire

Le code a été réorganisé en une architecture modulaire pour supporter plusieurs providers d'API Voice:

- **`voice_bridge.py`** - Classe abstraite `VoiceBridge` définissant l'interface commune
- **`openai_bridge.py`** - Implémentation pour l'API OpenAI Realtime
- **`grok_bridge.py`** - Implémentation pour l'API xAI Grok Voice Think Fast
- **`main.py`** - Point d'entrée avec sélection du provider via arguments ou variables d'environnement

## Différences clés entre les API

### OpenAI Realtime vs Grok Voice Think Fast

| Caractéristique | OpenAI Realtime | Grok Voice Think Fast |
| --------------- | --------------- | --------------------- |
| Endpoint | `wss://api.openai.com/v1/realtime` | `wss://api.x.ai/v1/realtime` |
| Auth | `OPENAI_API_KEY` | `XAI_API_KEY` |
| Modèles | gpt-realtime-mini, gpt-realtime, gpt-realtime-2 | grok-voice-think-fast-1.0 |
| Voix | alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar | eve, ara, rex, sal, leo |
| `conversation.item.truncate` | ✅ Supporté | ❌ Non supporté |
| `reasoning.effort` | ✅ Supporté (gpt-realtime, gpt-realtime-2) | ❌ Non supporté |
| Transcription input | gpt-4o-mini-transcribe | Auto-détection (20+ langues) |

### Impact du barge-in

Le barge-in (interruption quand l'utilisateur parle) fonctionne différemment:

- **OpenAI**: Annule la réponse ET tronque l'item conversationnel à ce qui a été joué
- **Grok**: Annule uniquement la réponse (pas de troncation)

Cela est géré automatiquement par la méthode `supports_truncate()` dans chaque bridge.

## Utilisation

### Configuration (.env)

```bash
# Provider: openai (défaut) ou xai
VOICE_PROVIDER=openai

# Clés API
OPENAI_API_KEY=sk-...
XAI_API_KEY=...

# Modèles
OPENAI_REALTIME_MODEL=gpt-realtime-mini
GROK_MODEL=grok-voice-think-fast-1.0

# Voix
OPENAI_REALTIME_VOICE=alloy
GROK_VOICE=eve
```

### Lancement

```bash
# OpenAI (par défaut)
./run.sh             # Utilise OPENAI_REALTIME_MODEL du .env
./run.sh mini        # gpt-realtime-mini
./run.sh full        # gpt-realtime
./run.sh full2       # gpt-realtime-2

# xAI Grok
./run.sh grok        # grok-voice-think-fast-1.0

# Custom
python main.py --provider openai --model gpt-realtime-2 --voice alloy
python main.py --provider xai --model grok-voice-think-fast-1.0 --voice eve
```

## Structure du code

### Classe VoiceBridge (abstraite)

Méthodes à implémenter par les sous-classes:

- `get_ws_url()` - URL WebSocket du provider
- `get_auth_header()` - Header d'authentification
- `get_session_config()` - Configuration de session
- `handle_audio_delta()` - Traitement audio entrant
- `handle_transcript_delta()` - Traitement transcription
- `handle_input_transcription()` - Traitement transcription input
- `supports_truncate()` - Support de conversation.item.truncate
- `compute_cost()` - Calcul des coûts
- `get_tools()` - Configuration des outils
- `_print_config()` - Affichage de la configuration

### Fonctionnalités communes

Gérées dans la classe de base `VoiceBridge`:

- Gestion WebSocket (connexion, envoi, réception)
- Stream audio microphone
- Barge-in (avec ou sans troncation selon le provider)
- Gestion des outils (play_emotion, look, move_sequence)
- Exécution des mouvements robot
- Playback des émotions
- Calcul des coûts

## Notes de compatibilité

L'API Grok Voice est compatible avec l'API OpenAI Realtime au niveau protocolaire, mais avec quelques différences:

1. **Événements non supportés par Grok**:
   - `conversation.item.truncate`
   - `conversation.item.retrieve`
   - `output_audio_buffer.clear`
   - Plusieurs événements de transcription détaillée

2. **Nommage des événements**:
   - Grok utilise `response.output_text.delta` au lieu de `response.audio_transcript.delta`

3. **Structure de session**:
   - Grok a une structure plus simple (pas de bloc `output_modalities`)
   - Pas de paramètre `reasoning.effort`

## Prix

Les prix pour Grok Voice Think Fast sont basés sur des estimations (à vérifier dans la documentation officielle xAI):

- Text input: $4.00 / 1M tokens
- Text output: $24.00 / 1M tokens
- Audio input: $32.00 / 1M tokens
- Audio output: $64.00 / 1M tokens

Ces valeurs sont configurables dans `grok_bridge.py` (dictionnaire `PRICING`).

## Tests recommandés

Avant de déployer en production:

1. Test avec `./run.sh grok` pour vérifier la connexion xAI
2. Test des mouvements simples (look left/right/up/down)
3. Test des chorégraphies (move_sequence)
4. Test du barge-in (parler pendant que le robot répond)
5. Comparaison des coûts entre OpenAI et xAI

## Dépannage

### Erreur "XAI_API_KEY is not set"
- Vérifier que `XAI_API_KEY` est défini dans `.env` si `VOICE_PROVIDER=xai`

### Erreur "Unsupported Grok voice"
- Les voix valides sont: eve, ara, rex, sal, leo

### Barge-in ne fonctionne pas comme attendu avec Grok
- Normal: Grok ne supporte pas la troncation, seulement l'annulation

### Transcription en anglais au lieu de français
- Grok détecte automatiquement la langue. Ajouter "Tu DOIS toujours répondre EN FRANÇAIS" dans les instructions.
