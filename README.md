# reachy-mini

Reachy Mini connecté à l'API **OpenAI Realtime** (`gpt-realtime-mini`)
en WebSocket, avec expression d'émotions sur le robot.

## Démarrage rapide

1. Installation complète (Python + GStreamer + plugin Rust webrtcsrc) :
   voir **[INSTALL.md](./INSTALL.md)**.
2. Configurer `.env` :
   ```bash
   cp .env.example .env
   # éditer : OPENAI_API_KEY, REACHY_HOST=<ip-LAN-du-robot>
   ```
3. Lancer :
   ```bash
   ./run.sh
   ```

`run.sh` pose `GST_PLUGIN_PATH` (plugin Rust) + `LD_PRELOAD`
(libstdc++/libgcc système, nécessaire si vous tournez sous miniconda)
et charge `.env`.

## Trouver l'IP du robot

- **Linux natif** : `ping -4 reachy-mini.local`
- **WSL2** : depuis PowerShell Windows, `ping -4 reachy-mini.local`
  (le mDNS WSL2 ne résout pas les `.local` par défaut)

## Émotions

Bibliothèque : `pollen-robotics/reachy-mini-emotions-library` (81 émotions).

| Évènement WS | Émotion jouée |
| --- | --- |
| `input_audio_buffer.speech_started` | `attentive1` |
| `input_audio_buffer.speech_stopped` | `thoughtful1` |
| `response.done` | `serenity1` |

Le modèle peut aussi appeler le tool `play_emotion(name)` pour exprimer
joie, surprise, curiosité, doute, etc. Liste exposée dans
`EMOTION_NAMES` (voir `main.py`).

## Voir aussi

- [INSTALL.md](./INSTALL.md) — installation complète + dépannage
- [OpenAI Realtime API docs](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
