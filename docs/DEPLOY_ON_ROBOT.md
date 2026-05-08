# Déployer le bridge sur le robot (mode autonome)

Ce document décrit comment exécuter le bridge `main.py` directement sur
le Raspberry Pi intégré au Reachy Mini Wireless, sans PC dans la boucle.

## Pourquoi

Dans le setup par défaut (`main.py` sur PC, `REACHY_HOST` sur le robot),
le PC lit les WAVs des émotions depuis son cache HuggingFace local et
streame les samples au robot via **WebRTC**. Le Pi du robot ne sert
qu'à faire tourner le daemon et à router le hardware.

En déployant le bridge sur le Pi :

- Les sons d'émotion sont lus depuis le cache HF du Pi (déjà chaud — le
  daemon le précharge au boot) et joués directement sur le speaker via
  GStreamer **local** (`alsasink`).
- Plus aucun byte audio sur le réseau pour les émotions → latence
  minimale.
- Le PC n'est plus nécessaire : alim + Wi-Fi + clé API suffisent.
- Le robot devient autonome (use case cible de la version wireless).

## Ce qui change automatiquement (et ne demande aucun changement de code)

Le SDK `reachy-mini` fait de l'auto-détection du backend média
(`reachy_mini.py`, fonction de résolution du backend) :

1. Si l'endpoint IPC du daemon est joignable en local → backend `LOCAL`.
2. Sinon → backend `WEBRTC`.

Donc le **même `main.py`** fonctionne :

- Sur PC, sans daemon local ⇒ `WEBRTC` (situation actuelle).
- Sur le Pi, avec daemon sur localhost ⇒ `LOCAL` (situation cible).

`mini.media.push_audio_sample(samples)` est routé différemment selon le
backend résolu, transparent pour l'`EmotionPlayer`.

`RecordedMoves("pollen-robotics/reachy-mini-emotions-library")` lira le
cache HF du Pi (peuplé par le daemon au boot), pas celui du PC.

## Étapes de déploiement

### 1. Préparer le bundle

Sur ton PC, vérifier que tout est bien committé. Le bundle à transférer
contient :

```
reachy-mini/
├── bridges/
├── scripts/
├── main.py
├── requirements.txt
└── .env          # OPENAI_API_KEY=…  (sans REACHY_HOST)
```

`.env` doit contenir uniquement :

```
OPENAI_API_KEY=sk-…
# OPENAI_REALTIME_MODEL=gpt-realtime-mini   # facultatif
# VOICE_PROVIDER=openai                     # facultatif
```

Pas de `REACHY_HOST` : le SDK utilisera `localhost` par défaut.

### 2. Copier sur le Pi

```bash
ssh pi@reachy-mini.local "mkdir -p ~/voice-bridge"
scp -r bridges scripts main.py requirements.txt .env pi@reachy-mini.local:~/voice-bridge/
```

(Le hostname exact peut varier ; remplacer par l'IP LAN si mDNS ne
résout pas.)

### 3. Installer les dépendances dans un venv

```bash
ssh pi@reachy-mini.local
cd ~/voice-bridge
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`reachy-mini` est probablement déjà installé en système (le daemon
l'utilise). Le venv aura sa propre copie — c'est plus propre et évite
les conflits si un jour tu mets à jour le bridge sans toucher au
daemon.

> ⚠️ Si le venv installe une version de `reachy-mini` plus récente que
> celle du daemon, vérifier la compatibilité de l'API client/serveur.
> Au pire, pinner la version :
> `pip install reachy-mini==<version-du-daemon>`.

### 4. Test en interactif

```bash
cd ~/voice-bridge
set -a; source .env; set +a
.venv/bin/python main.py
```

Au démarrage, on doit voir dans les logs :

- `[ws] connected` (côté OpenAI/Grok)
- `[emotion] preloaded N sounds (X.Y MB @ R Hz), Z silent` (depuis le
  cache HF du Pi)
- Pas de message WebRTC (le SDK a basculé en `LOCAL`).

Parler au robot → l'émotion déclenchée doit jouer **directement** sur
son speaker, sans relais réseau.

### 5. Auto-start avec systemd

Pour que le bridge se relance automatiquement au boot du robot :

`/etc/systemd/system/voice-bridge.service` :

```ini
[Unit]
Description=Reachy Mini Voice Bridge
After=network-online.target reachy-mini-daemon.service
Wants=network-online.target
Requires=reachy-mini-daemon.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/voice-bridge
EnvironmentFile=/home/pi/voice-bridge/.env
ExecStart=/home/pi/voice-bridge/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Activation :

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-bridge
journalctl -u voice-bridge -f   # logs en live
```

Reboot du robot → le bridge redémarre tout seul.

> Le nom exact du service daemon (`reachy-mini-daemon.service`) est à
> confirmer avec `systemctl list-units --type=service | grep -i reachy`
> sur le Pi.

## Caveats

- **Préchargement HF** : si systemd démarre `voice-bridge` trop vite
  après le daemon, le cache HF pourrait ne pas encore être chaud, et
  `RecordedMoves` retombera sur un download réseau (lent au premier
  boot). Le `Requires=reachy-mini-daemon.service` ordonne le démarrage,
  mais ne garantit pas que le daemon a terminé son init. Si problème,
  ajouter un check de readiness dans un `ExecStartPre=` ou un sleep
  initial.

- **Compute du Pi** : le bridge est I/O-bound (WebSocket cloud + audio
  streams). Devrait passer large sur Pi 4 / 5. À surveiller : ajouter
  Whisper local, vision, reasoning lourd ⇒ ça pourrait serrer.

- **Sécurité du `.env`** : `OPENAI_API_KEY` traîne en clair sur la SD du
  robot. Si tu prêtes le robot ou si la SD est compromise, faire une
  rotation de la clé. Pour durcir, utiliser une variante systemd
  `LoadCredential=` ou un secret store.

- **Mises à jour** : pour mettre à jour le bridge, `scp` la nouvelle
  version par-dessus puis `sudo systemctl restart voice-bridge`. Pas de
  CI/CD ici — flow manuel.

- **Debugging plus pénible que sur PC** : moins d'outils, logs via
  `journalctl`, pas d'éditeur agréable. Itérer sur PC d'abord, déployer
  une fois stabilisé.

## Pour info : déployer le script d'export

`scripts/export_emotion_sounds.py` peut aussi tourner sur le Pi pour
copier les WAVs vers une clé USB par exemple :

```bash
.venv/bin/python scripts/export_emotion_sounds.py /media/usb/emotion_sounds/
```

Pas de download réseau (cache HF déjà sur le Pi), juste de la copie
locale.
