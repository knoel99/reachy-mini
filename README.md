# reachy-mini

Reachy Mini connecté aux **API Voice** (OpenAI Realtime ou xAI Grok Voice Think Fast)
en WebSocket, avec mouvements de tête, chorégraphies planifiées et expressions
d'émotions sur le robot.

## Architecture et flux de l'application

### Flux complet : de la voix utilisateur à la réponse du robot

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UTILISATEUR                                      │
│  Parle au microphone de Reachy Mini                                         │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REACHY MINI (Matériel)                                    │
│  Microphone → Échantillonnage audio (ex: 48000 Hz)                          │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BRIDGE LOCAL (Python)                                     │
│  1. Capture audio via `media.get_audio_sample()`                            │
│  2. Conversion en mono si nécessaire                                         │
│  3. Rééchantillonnage vers 24000 Hz (taux API)                              │
│  4. Conversion float32 → PCM16                                               │
│  5. Encodage en base64                                                       │
│  6. Envoi WebSocket: `input_audio_buffer.append`                             │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    API VOICE (OpenAI / xAI)                                  │
│  1. Détection VAD (Voice Activity Detection) - serveur_vad                  │
│  2. Speech-to-Text (STT) - transcription en temps réel                       │
│  3. Reasoning - Le modèle comprend et planifie la réponse                    │
│  4. Function Calling - Le modèle peut invoquer des outils:                  │
│     • `play_emotion(name)` - Jouer une émotion                               │
│     • `look(direction)` - Tourner la tête                                    │
│     • `move_sequence(steps)` - Chorégraphie complexe                         │
│  5. Text-to-Speech (TTS) - Génération audio de la réponse                   │
│  6. Streaming audio via WebSocket: `response.output_audio.delta`             │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BRIDGE LOCAL (Python)                                     │
│  1. Réception WebSocket des deltas audio                                    │
│  2. Décodage base64 → PCM16                                                  │
│  3. Conversion PCM16 → float32                                               │
│  4. Rééchantillonnage vers taux du robot (ex: 48000 Hz)                      │
│  5. Push vers le media manager: `media.push_audio_sample()`                   │
│  6. Affichage transcription en temps réel                                    │
│  7. Gestion des tools (émotions, mouvements)                                 │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REACHY MINI (Matériel)                                    │
│  1. Lecture audio via haut-parleur                                          │
│  2. Exécution des mouvements (tête, antennes)                               │
│  3. Playback des émotions préenregistrées                                    │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UTILISATEUR                                      │
│  Entend la réponse vocale du robot + voit les mouvements                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gestion du barge-in (interruption)

```
Utilisateur parle pendant la réponse du robot
         │
         ▼
┌─────────────────────────────────────────────┐
│  API détecte `input_audio_buffer.speech_started` │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Bridge local envoie `response.cancel`      │
│  + (OpenAI uniquement) `conversation.item.truncate` │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Arrêt immédiat de la réponse en cours     │
│  L'API traite la nouvelle entrée utilisateur│
└─────────────────────────────────────────────┘
```

### Gestion des tools (Function Calling)

```
Modèle IA décide d'utiliser un tool
         │
         ▼
┌─────────────────────────────────────────────┐
│  Événement `response.output_item.done`     │
│  avec type="function_call"                 │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Bridge exécute localement:                │
│  • play_emotion → Playback émotion HF      │
│  • look → goto_target(pose)                │
│  • move_sequence → Séquence de poses       │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Envoi résultat via `conversation.item.create`│
│  avec type="function_call_output"          │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Modèle IA réagit au résultat et continue  │
│  la conversation (ou génère audio final)   │
└─────────────────────────────────────────────┘
```

### Différences OpenAI vs xAI

| Étape | OpenAI Realtime | xAI Grok Voice |
|-------|-----------------|----------------|
| Barge-in | Cancel + Truncate | Cancel uniquement |
| Événement transcription | `response.audio_transcript.delta` | `response.output_text.delta` |
| Reasoning | Supporte `reasoning.effort` | Non supporté |
| Session structure | Complète avec `output_modalities` | Simplifiée |

## Démarrage rapide

1. Installation complète (Python + GStreamer + plugin Rust webrtcsrc) :
   voir **[INSTALL.md](./INSTALL.md)**.
2. Configurer `.env` :
   ```bash
   cp .env.example .env
   # éditer : OPENAI_API_KEY ou XAI_API_KEY, REACHY_HOST=<ip-LAN-du-robot>
   ```
3. Lancer :
   ```bash
   # OpenAI (par défaut)
   ./run.sh             # provider=openai (lit OPENAI_REALTIME_MODEL)
   ./run.sh mini        # provider=openai model=gpt-realtime-mini
   ./run.sh full        # provider=openai model=gpt-realtime
   ./run.sh full2       # provider=openai model=gpt-realtime-2
   
   # xAI Grok Voice
   ./run.sh grok        # provider=xai model=grok-voice-think-fast-1.0
   
   # Custom
   ./run.sh provider et du openai <model>
   ./run.sh xai <model>
### OpenAI Realtime

`run.sh` pose `GST_PLUGIN_PATH` (plugin Rust) + `LD_PRELOAD`
(libstdc++/libgcc système, nécessaire si vous tournez sous miniconda)
et charge `.env`.ations     |

### xAI Grok Voice Think Fast

| Modèle                      | Voix disponibles      | Remrques                                                    |
| --------------------------- | --------------------- | ------------------------------------------------------------ |
| `grok-voice-hnk-fast-1.0` | eve, ara, rex, sal, leo | Modèle récent xAI. Cmpatible OpeAI Realtime API. Ne upportepas`conversation.item.truncate`(barge-insimplifié).

## Choix du modèle

| Modèle              | Audio in/out / 1M | `reasoning.effort` | Recommandé pour                                              |
| ------------------- | ----------------- | ------------------ | ------------------------------------------------------------ |
| `gpt-realtime-mini` | $10 / $20         | non supporté       | Conversation simple, faible coût                             |
| `gpt-realtime`      | $32 / $64         | supporté           | Plus de tokens contexte, meilleur tool calling               |
| `gpt-realtime-2`    | $32 / $64         | supporté           | **Latest**. Plan multi-étapes, chorégraphies, imitations     |

`gpt-realtime-2` est commercialisé par OpenAI comme un *« reasoning voice
agent »* qui peut « *think before it speaks* » et fait du *« interleaved
thinking between tool calls »* — c'est notre cible pour les chorégraphies
complexes (imite une poule, dessine un cercle, danse…).

`reasoning.effort` accepte `minimal | low | medium | high`. Le pont
défaut sur `medium` (configurable via `OPENAI_REASONING_EFFORT`).
Plus haut = meilleur plan mais plus de latence et de tokens.

À chaque tour le pont affiche `cost=$X cumul=$Y` et la session se termine
sur `[cost] session total: $Z over N turn(s)`. Au démarrage une bannière
`[config] model=... reasoning.effort=... prices /1M tok: ...` rappelle
le tarif et les réglages en cours.

## Trouver l'IP du robot

- **Linux natif** : `ping -4 reachy-mini.local`
- **WSL2** : depuis PowerShell Windows, `ping -4 reachy-mini.local`
  (le mDNS WSL2 ne résout pas les `.local` par défaut)

## Tools exposés au modèle

Le modèle invoque ces fonctions via le mécanisme function-calling de
l'API Realtime (jamais en texte) :

- **`play_emotion(name)`** — joue une émotion préenregistrée (joie,
  surprise, curiosité, doute…). 34 émotions issues du dataset HF
  `pollen-robotics/reachy-mini-emotions-library`. L'émotion appelée
  pendant que le bot parle est mise en file et jouée à `response.done`
  pour ne pas couvrir la voix.
- **`look(direction)`** — tourne la tête vers `left`, `right`, `up`,
  `down`, `center`. Bouge en parallèle de la voix (silencieux).
- **`move_sequence(steps)`** — chorégraphie planifiée. Chaque step
  contient `yaw`, `pitch`, `roll` (degrés), optionnellement
  `antenna_left`/`antenna_right` (degrés) et `duration` (secondes).
  À utiliser pour : faire un cercle de tête, hocher, secouer, danser,
  imiter un animal, explorer du regard.

Barge-in actif : si tu parles pendant que le robot répond, le pont
envoie `response.cancel` + `conversation.item.truncate` pour que le
robot s'arrête immédiatement.

## Voir aussi

- [INSTALL.md](./INSTALL.md) — installation complète + dépannage
- [docs/GROK_INTEGRATION.md](./docs/GROK_INTEGRATION.md) — documentation de l'intégration Grok Voice
- [bridges/README.md](./bridges/README.md) — documentation des bridges API
- [OpenAI Realtime API docs](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [xAI Voice Agent API docs](https://docs.x.ai/developers/model-capabilities/audio/voice-agent)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
