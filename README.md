# reachy-mini

Reachy Mini connecté à l'**API OpenAI Realtime** (`gpt-realtime-mini` ou
`gpt-realtime`) en WebSocket, avec mouvements de tête, chorégraphies
planifiées et expressions d'émotions sur le robot.

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
   ./run.sh             # modèle par défaut (lit OPENAI_REALTIME_MODEL)
   ./run.sh mini        # force gpt-realtime-mini
   ./run.sh full        # force gpt-realtime (plus cher, meilleur)
   ```

`run.sh` pose `GST_PLUGIN_PATH` (plugin Rust) + `LD_PRELOAD`
(libstdc++/libgcc système, nécessaire si vous tournez sous miniconda)
et charge `.env`.

## Choix du modèle

| Modèle              | Coût audio in / out (par 1M tok) | Forces                                                    |
| ------------------- | -------------------------------- | --------------------------------------------------------- |
| `gpt-realtime-mini` | $10 / $20                        | Pas cher, conversation fluide, tool calls simples         |
| `gpt-realtime`      | $32 / $64                        | Bien meilleur en planification, tool calls multi-étapes,  |
|                     |                                  | suivi d'instructions complexes (chorégraphies, imitation) |

À chaque tour le pont affiche `cost=$X cumul=$Y` et la session se termine
sur `[cost] session total: $Z over N turn(s)`. Au démarrage une bannière
`[config] model=... prices /1M tok: ...` rappelle le tarif en cours.

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
- [OpenAI Realtime API docs](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
