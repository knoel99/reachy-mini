# reachy-mini

Reachy Mini driven by an LLM through one of two voice pipelines: **OpenAI
Realtime** (bidirectional audio) or **Grok chat-with-tools** (mic → STT →
chat with function-calling). In both cases the robot has no voice — it
only reacts through tool calls: head movements, antennas, choreographies,
preloaded emotion sounds.

## Layout

```
reachy-mini/
├── pyproject.toml
├── run.sh                        # loads .env, GST_PLUGIN_PATH, LD_PRELOAD
├── docs/DEPLOY_ON_ROBOT.md       # systemd deployment on the Pi
├── scripts/export_emotion_sounds.py
├── colab/fastvlm_7b_server.ipynb # FastVLM-7B server (Colab GPU)
├── src/reachy_voice/             # voice pipeline
│   ├── __main__.py               # python -m reachy_voice
│   ├── _actions.py               # RobotActions: tool → robot dispatch (shared)
│   ├── _log.py                   # log() with timestamp + delta
│   ├── emotions.py               # EmotionPlayer (preload + push_audio_sample)
│   ├── melody.py                 # MelodyPlayer (sine-wave melody synth)
│   ├── tools.py                  # INSTRUCTIONS, LOOK_POSES, build_tools
│   ├── openai/                   # OpenAI provider
│   │   └── realtime.py           # OpenAIRealtimeBridge (WebSocket)
│   └── grok/                     # Grok provider
│       ├── vad.py                # webrtcvad: utterance segmentation
│       ├── stt.py                # POST /v1/stt (xAI Speech-to-Text)
│       └── chat.py               # GrokChatBridge: orchestrator
└── src/reachy_vision/            # vision package (optional)
    ├── camera.py                 # Camera: latest BGR frame under lock
    ├── moondream.py              # Moondream: caption() + point()
    ├── fastvlm.py                # FastVLM: caption() (HTTP to Colab)
    └── preview.py                # Preview: MJPEG :5050 + debug overlays
```

Install: `pip install -e .` (see [INSTALL.md](./INSTALL.md)).
For vision: `pip install -e '.[vision]'` (adds Moondream).
Run: `./run.sh <model>`, `python -m reachy_voice`, or `reachy-voice`.

### Enabling vision (optional)

Copy `.env.example` to `.env` then configure a backend:

```bash
# Option A — Moondream cloud (recommended to start)
VISION_BACKEND=moondream
MOONDREAM_API_KEY=...    # https://moondream.ai/c/account/api-keys

# Option B — FastVLM-7B served by Colab (see colab/fastvlm_7b_server.ipynb)
VISION_BACKEND=fastvlm
FASTVLM_URL=https://xxxx.trycloudflare.com
```

Two extra tools are then exposed to the LLM:
- `look_and_describe(question)` — capture a frame and ask a question
- `find_object(target)` — locate an object and aim the head
  (Moondream only — uses native `point()`)

A debug web preview runs at `http://localhost:5050` as soon as the
vision backend starts. It streams live MJPEG and overlays the latest
caption + the crosshair of the most recent `find_object` (each overlay
fades after 5 s). Disable with `VISION_PREVIEW=0`. The preview binds
on `0.0.0.0` — firewall your LAN or disable it in prod on the Pi.

## The two pipelines

### OpenAI Realtime (`provider=openai`)

```
mic → PCM 24 kHz → WebSocket → server VAD → STT → model (streaming
tool calls) → response.done → dispatch on the robot
```

`output_modalities=["text"]` disables TTS. Only `gpt-realtime-2`
supports interleaved reasoning (`reasoning.effort`).

### Grok chat-with-tools (`provider=grok`)

```
mic → PCM 16 kHz → local webrtcvad → utterance → POST /v1/stt →
text → POST /v1/chat/completions (tools) → tool_calls → dispatch →
re-call chat with results → loop
```

No realtime API. Roughly 100× cheaper per turn for tool-only usage,
with more predictable latency (no TTS budget wasted). Native model
context, no manual truncation.

## Getting started

1. **Install** (Python + GStreamer + Rust `webrtcsrc` plugin): see
   **[INSTALL.md](./INSTALL.md)**.
2. **Configure `.env`**:
   ```bash
   cp .env.example .env
   # edit: OPENAI_API_KEY or XAI_API_KEY, REACHY_HOST=<robot-LAN-ip>
   ```
3. **Run** — the shortcut is the full model name; the provider is
   inferred from the prefix:
   ```bash
   # OpenAI Realtime
   ./run.sh gpt-realtime-mini
   ./run.sh gpt-realtime
   ./run.sh gpt-realtime-2

   # Grok chat-with-tools
   ./run.sh grok-4-1-fast-non-reasoning
   ./run.sh grok-4-1-fast-reasoning

   # Explicit override if needed
   ./run.sh openai <model>
   ./run.sh grok   <model>

   # No argument: defaults from .env
   ./run.sh
   ```

## Picking a model

### OpenAI Realtime

| Model               | Audio in/out / 1M | reasoning.effort | Notes                                          |
| ------------------- | ----------------- | ---------------- | ---------------------------------------------- |
| `gpt-realtime-mini` | $10 / $20         | no               | Cheap, simple conversation                     |
| `gpt-realtime`      | $32 / $64         | no               | Solid tool-calling                             |
| `gpt-realtime-2`    | $32 / $64         | yes              | **Latest**, multi-step planning (choreography) |

`reasoning.effort` ∈ `minimal | low | medium | high`, defaults to
`medium`, configurable via `OPENAI_REASONING_EFFORT`.

### Grok chat

| Model                         | Input / 1M | Output / 1M | Notes                          |
| ----------------------------- | ---------- | ----------- | ------------------------------ |
| `grok-4-1-fast-non-reasoning` | $0.20      | $0.50       | Fastest, default               |
| `grok-4-1-fast-reasoning`     | $0.20      | $0.50       | Planning for complex sequences |

Plus REST STT: **$0.10/hr** of transcribed audio.

## Logs

Every event line is prefixed with `[HH:MM:SS.mmm +Δs]` (delta from the
previous log — handy for measuring per-step latency).

At startup: `[config] provider=… model=… prices …`. Per turn:
`cost=$X cumul=$Y` plus token breakdown. At shutdown: `[cost] session
total: $Z over N turn(s)`.

## Tools exposed to the model

The model invokes these functions via function-calling (never as text):

- **`play_emotion(name)`** — preloaded emotion (movement + bundled
  audio). Enum populated dynamically from the HF dataset
  `pollen-robotics/reachy-mini-emotions-library` (~80 emotions). The
  bundled choreography is time-stretched to match the audio so head
  motion and sound stay in sync.
- **`look(direction)`** — head toward `left`, `right`, `up`, `down`,
  `center`.
- **`move_sequence(steps)`** — planned choreography. Each step:
  `yaw`, `pitch`, `roll` (deg), optionally `antenna_left` /
  `antenna_right` (deg) and `duration` (s). For circles, nods,
  dances, imitations.

All three tools are implemented in
[`_actions.py`](./src/reachy_voice/_actions.py) and shared across both
providers. Calls are serialised through a single worker so motors and
speaker never collide.

## Finding the robot's IP

- **Native Linux**: `ping -4 reachy-mini.local`
- **WSL2**: from Windows PowerShell, `ping -4 reachy-mini.local`
  (WSL2's mDNS does not resolve `.local` by default)

## See also

- [INSTALL.md](./INSTALL.md) — installation + troubleshooting
- [docs/DEPLOY_ON_ROBOT.md](./docs/DEPLOY_ON_ROBOT.md) — systemd deployment on the Pi
- [OpenAI Realtime API](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [xAI Voice Agent API](https://docs.x.ai/developers/model-capabilities/audio/voice-agent)
- [xAI Speech-to-Text REST](https://docs.x.ai/developers/rest-api-reference/inference/voice)
- [xAI Models & Pricing](https://docs.x.ai/developers/models)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
