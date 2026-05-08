"""Abstract base class for voice API bridges (OpenAI Realtime / Grok Voice)."""

from __future__ import annotations

import abc
import json
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import websocket

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


# ---- Shared constants ----

REALTIME_RATE = 24_000

INSTRUCTIONS = """# Rôle & Objectif
Tu es la voix d'un petit robot de bureau Reachy Mini. Tu n'es PAS un
chatbot vocal ordinaire : tu es un AGENT PLANIFICATEUR qui pilote un
corps physique. Tu réfléchis avant d'agir.

# Personnalité & Ton
Chaleureux, curieux, expressif. Bref : 1 à 2 phrases par tour.

# Langue
Tu DOIS toujours répondre EN FRANÇAIS, jamais dans une autre langue,
même si l'utilisateur s'adresse à toi dans une autre langue.

# Contexte du corps
Tu disposes :
- d'une tête articulée. Elle a SIX degrés de liberté :
    * rotations : yaw ±60°, pitch ±30°, roll ±30°
    * translations : x ±30 mm (avant/arrière), y ±30 mm (gauche/droite),
      z ±30 mm (haut/bas — fait littéralement MONTER ou descendre la tête)
  ATTENTION : `pitch` ≠ `z`. Pitch lève le MENTON. Z élève toute la tête.
  Si l'utilisateur dit « monte la tête », « élève la tête » ou « tête en
  hauteur », c'est `z` positif, PAS pitch.
- de deux antennes mobiles,
- d'une bibliothèque d'émotions préenregistrées.

# Outils
Tu as TROIS outils, à invoquer UNIQUEMENT via le mécanisme de
function-calling de l'API (jamais en texte parlé).

- `play_emotion(name)` — joue une émotion préenregistrée. L'émotion
  est jouée à la fin de ton tour de parole pour ne pas couvrir la voix.
- `look(direction)` — tourne la tête vers UNE direction simple :
  left, right, up, down, center. À utiliser uniquement pour un
  mouvement statique.
- `move_sequence(steps, archetype?)` — chorégraphie planifiée. À
  utiliser pour TOUT mouvement composé ou dynamique : cercle, hochement,
  secouement, danse, imitation d'animal, exploration du regard…
  Tu PLANIFIES la séquence en émettant 6 à 20 keyframes (yaw/pitch/roll
  en degrés + durée). Renseigne `archetype` quand l'intention rentre
  dans un pattern connu (`nod`, `shake`, `circle`, `figure_eight`,
  `dance`, `mime`, `explore`).

# Règles
- Ne réponds JAMAIS « je ne peux pas bouger » — tu peux toujours.
  Si la demande est complexe, planifie-la dans `move_sequence`.
- Pour toute demande de forme géométrique, danse ou imitation
  (cercle, infini, danse, poule, chat…), émets UN appel
  `move_sequence` avec ≥ 6 keyframes pour que ce soit lisible.
- Les noms d'outils ne doivent JAMAIS apparaître dans ton audio. Si
  tu te vois prononcer « play_emotion », « look » ou « move_sequence »
  c'est une ERREUR.

# Flux de conversation
1. L'utilisateur parle → tu écoutes.
2. Si la demande est SIMPLE (saluer, répondre, expression spontanée) :
   tu réponds en 1-2 phrases et tu peux appeler `play_emotion` et/ou
   `look` en parallèle.
3. Si la demande est COMPLEXE (chorégraphie, imitation, séquence) :
   tu dis une COURTE PRÉAMBULE parlée (« laisse-moi imaginer ça… »
   ou « ok, je planifie »), PUIS tu émets l'appel `move_sequence`
   avec la chorégraphie planifiée.
"""

EMOTION_NAMES = [
    "amazed1", "anxiety1", "attentive1", "attentive2", "calming1",
    "cheerful1", "confused1", "curious1", "displeased1", "enthusiastic1",
    "exhausted1", "frustrated1", "grateful1", "helpful1", "inquiring1",
    "irritated1", "laughing1", "loving1", "no1", "oops1", "proud1",
    "relief1", "sad1", "scared1", "serenity1", "shy1", "success1",
    "surprised1", "thoughtful1", "tired1", "uncertain1",
    "understanding1", "welcoming1", "yes1",
]


def _make_head_pose(roll_deg: float = 0.0, pitch_deg: float = 0.0,
                    yaw_deg: float = 0.0,
                    x_mm: float = 0.0, y_mm: float = 0.0,
                    z_mm: float = 0.0) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix for head pose."""
    from scipy.spatial.transform import Rotation as R
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
    ).as_matrix()
    pose[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return pose


LOOK_POSES = {
    "center": _make_head_pose(),
    "left":   _make_head_pose(yaw_deg=30),
    "right":  _make_head_pose(yaw_deg=-30),
    "up":     _make_head_pose(pitch_deg=-20),
    "down":   _make_head_pose(pitch_deg=20),
}


TOOLS = [
    {
        "type": "function",
        "name": "play_emotion",
        "description": (
            "Joue une émotion physique sur le robot Reachy Mini "
            "(mouvements de tête + antennes). À utiliser quand une émotion "
            "renforce naturellement la réponse. L'émotion est jouée à la "
            "fin du tour de parole pour ne pas couvrir la voix."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nom de l'émotion à jouer.",
                    "enum": EMOTION_NAMES,
                },
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "look",
        "description": (
            "Tourne la tête de Reachy Mini dans une direction simple. "
            "Pour un mouvement composé (cercle, danse, imitation…), "
            "utiliser plutôt `move_sequence`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Direction où regarder.",
                    "enum": list(LOOK_POSES.keys()),
                },
            },
            "required": ["direction"],
        },
    },
    {
        "type": "function",
        "name": "move_sequence",
        "description": (
            "Joue une chorégraphie de la tête planifiée par toi. À "
            "UTILISER pour tout mouvement composé ou dynamique : "
            "cercle, figure en huit, hochement (oui), secouement (non), "
            "danse, imitation d'animal, regard exploratoire. Émets "
            "ENTRE 6 ET 20 keyframes pour que la chorégraphie soit "
            "lisible. Exemples concrets :\n"
            "- 'hocher la tête' (oui) : pitch alterne -15/+15 sur 4-6 steps.\n"
            "- 'secouer la tête' (non) : yaw alterne -25/+25 sur 4-6 steps.\n"
            "- 'cercle de tête' : 8-12 keyframes sur un cercle yaw=cos*30,"
            " pitch=sin*15.\n"
            "- 'imiter une poule' : pitch -15→+25 répété + petits yaws +"
            " antennes qui frémissent.\n"
            "- 'danser' : combiner yaw/roll/antennes au rythme, 12-20"
            " keyframes.\n"
            "Le robot revient au neutre automatiquement à la fin."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "archetype": {
                    "type": "string",
                    "description": "Catégorie de l'intention. Aide le modèle à planifier des keyframes pertinentes. Optionnel.",
                    "enum": ["nod", "shake", "circle", "figure_eight",
                             "dance", "mime", "explore", "custom"],
                },
                "steps": {
                    "type": "array",
                    "description": "Suite ordonnée de poses cibles (6 à 20 keyframes pour les mouvements lisibles).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "yaw":   {"type": "number",
                                      "description": "Rotation yaw en degrés (-60..60). Positif=gauche, négatif=droite."},
                            "pitch": {"type": "number",
                                      "description": "Rotation pitch en degrés (-30..30). Positif=bas, négatif=haut (lève le menton). N'est PAS le fait d'élever physiquement la tête — pour ça, utiliser z."},
                            "roll":  {"type": "number",
                                      "description": "Rotation roll (penché côté) en degrés (-30..30)."},
                            "x":     {"type": "number",
                                      "description": "Translation X en millimètres (-30..30). Positif=avant. Sert à pencher la tête en avant."},
                            "y":     {"type": "number",
                                      "description": "Translation Y en millimètres (-30..30). Positif=gauche."},
                            "z":     {"type": "number",
                                      "description": "Translation Z en millimètres (-30..30). Positif=HAUT — fait MONTER la tête physiquement (le buste de la tête monte). C'est différent du pitch (qui ne fait que lever le menton)."},
                            "antenna_left":  {"type": "number",
                                              "description": "Antenne gauche en degrés (-90..90). Optionnel."},
                            "antenna_right": {"type": "number",
                                              "description": "Antenne droite en degrés (-90..90). Optionnel."},
                            "duration": {"type": "number",
                                         "description": "Durée pour atteindre cette pose en secondes (0.1..3.0). Pour un mouvement rapide rythmé, utiliser ~0.2-0.3 ; pour un mouvement lent expressif, ~0.6-1.5."},
                        },
                        "required": ["duration"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
]


class EmotionPlayer:
    """Serializes emotion playback. Skips silently if one is already playing."""
    
    def __init__(self, mini: ReachyMini, library) -> None:
        self.mini = mini
        self.library = library
        self._busy = threading.Lock()
        self._last_t = 0.0
    
    def play(self, name: str, min_interval: float = 2.5) -> bool:
        now = time.monotonic()
        if (now - self._last_t) < min_interval:
            return False
        if not self._busy.acquire(blocking=False):
            return False
        
        move = self.library.get(name)
        if move is None:
            self._busy.release()
            print(f"[emotion] unknown: {name}", flush=True)
            return False
        
        self._last_t = now
        
        def _run() -> None:
            try:
                self.mini.play_move(move, initial_goto_duration=0.5)
            except Exception as e:
                print(f"[emotion] play '{name}' failed: {e}", flush=True)
            finally:
                self._busy.release()
        
        threading.Thread(target=_run, daemon=True).start()
        return True


# ---- Abstract VoiceBridge class ----

class VoiceBridge(abc.ABC):
    """Abstract base class for voice API WebSocket bridges.
    
    Subclasses implement provider-specific logic (OpenAI vs xAI/Grok).
    """
    
    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self.media = mini.media
        self.in_rate = self.media.get_input_audio_samplerate()
        self.out_rate = self.media.get_output_audio_samplerate()
        
        self._ws = None  # type: websocket.WebSocketApp | None
        self._send_q = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._sender_thread = None  # type: threading.Thread | None
        self._mic_thread = None  # type: threading.Thread | None
        self._move_thread = None  # type: threading.Thread | None
        
        # Response state
        self._response_active = False
        self._speaking = False
        self._current_item_id = None
        self._current_content_index = 0
        self._audio_bytes_in_response = 0
        self._first_delta_t = None
        self._audio_chunks = 0
        self._last_done_response_id = None
        self._needs_followup_response = False
        self._pending_emotion = None
        
        # Cost tracking
        self._cost_total = 0.0
        self._turns = 0
    
    # ---- Abstract methods (provider-specific) ----
    @abc.abstractmethod
    def get_ws_url(self) -> str:
        """Return WebSocket URL for the provider."""
        pass
    
    @abc.abstractmethod
    def get_auth_header(self) -> str:
        """Return authorization header for the provider."""
        pass
    
    @abc.abstractmethod
    def get_session_config(self) -> dict:
        """Return session configuration for the provider."""
        pass
    
    @abc.abstractmethod
    def handle_audio_delta(self, evt: dict) -> None:
        """Handle incoming audio delta from provider."""
        pass
    
    @abc.abstractmethod
    def handle_transcript_delta(self, evt: dict) -> None:
        """Handle transcript delta from provider."""
        pass
    
    @abc.abstractmethod
    def handle_input_transcription(self, evt: dict) -> None:
        """Handle input audio transcription completion."""
        pass
    
    @abc.abstractmethod
    def supports_truncate(self) -> bool:
        """Return True if provider supports conversation.item.truncate."""
        pass
    
    @abc.abstractmethod
    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        """Compute cost from usage block. Returns (cost_usd, breakdown)."""
        pass
    
    @abc.abstractmethod
    def get_tools(self) -> list[dict]:
        """Return the tools configuration for the provider."""
        pass
    
    @abc.abstractmethod
    def _print_config(self) -> None:
        """Print configuration banner."""
        pass
    
    # ---- Common helpers ----
    def _send(self, payload: dict) -> None:
        try:
            self._send_q.put_nowait(json.dumps(payload))
        except queue.Full:
            pass
    
    def _reset_response_state(self) -> None:
        self._response_active = False
        self._speaking = False
        self._current_item_id = None
        self._current_content_index = 0
        self._audio_bytes_in_response = 0
        self._first_delta_t = None
        self._audio_chunks = 0
    
    # ---- Audio helpers (shared) ----
    @staticmethod
    def f32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype("<i2").tobytes()
    
    @staticmethod
    def pcm16_bytes_to_f32(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
    
    @staticmethod
    def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return samples
        from scipy.signal import resample_poly
        g = np.gcd(src_rate, dst_rate)
        return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)
    
    # ---- barge-in (provider-specific) ----
    def _barge_in(self) -> None:
        """User started speaking. Cancel in-flight response."""
        if not self._response_active:
            return
        
        if self.supports_truncate():
            self._barge_in_with_truncate()
        else:
            self._barge_in_simple()
        
        self._pending_emotion = None
    
    def _barge_in_with_truncate(self) -> None:
        """Barge-in with conversation.item.truncate (OpenAI)."""
        recv_ms = self._audio_bytes_in_response // 48  # PCM16 mono @ 24kHz -> 48 bytes/ms
        wall_ms = 0
        if self._first_delta_t is not None:
            wall_ms = int((time.monotonic() - self._first_delta_t) * 1000)
        played_ms = min(recv_ms, wall_ms) if wall_ms > 0 else recv_ms
        
        if self._current_item_id and played_ms > 0:
            self._send({
                "type": "conversation.item.truncate",
                "item_id": self._current_item_id,
                "content_index": self._current_content_index,
                "audio_end_ms": played_ms,
            })
        self._send({"type": "response.cancel"})
        print(f"[barge-in] cut at {played_ms} ms", flush=True)
    
    def _barge_in_simple(self) -> None:
        """Barge-in without truncate support (Grok)."""
        self._send({"type": "response.cancel"})
        print(f"[barge-in] canceled (truncate not supported)", flush=True)
    
    # ---- WebSocket lifecycle ----
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        print("[ws] connected", flush=True)
        ws.send(json.dumps({"type": "session.update", "session": self.get_session_config()}))
        self.media.start_playing()
        self.media.start_recording()
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()
    
    def _mic_loop(self) -> None:
        """Stream microphone audio to the provider."""
        import base64
        
        while not self._stop.is_set():
            samples = self.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            mono = samples if samples.ndim == 1 else samples.mean(axis=1)
            resampled = self.resample(mono.astype(np.float32), self.in_rate, REALTIME_RATE)
            b64 = base64.b64encode(self.f32_to_pcm16_bytes(resampled)).decode("ascii")
            self._send({"type": "input_audio_buffer.append", "audio": b64})
    
    def _sender_loop(self) -> None:
        """Send queued messages to WebSocket."""
        while not self._stop.is_set():
            try:
                msg = self._send_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._ws.send(msg)  # type: ignore[union-attr]
            except Exception as e:
                print(f"[ws] send failed: {e}", flush=True)
                return
    
    def _on_error(self, ws: websocket.WebSocketApp, err: Exception) -> None:
        print(f"[ws] error: {err}", flush=True)
    
    def _on_close(self, ws, code, msg) -> None:
        print(f"[ws] closed code={code} msg={msg}", flush=True)
        print(f"[cost] session total: ${self._cost_total:.4f} over {self._turns} turn(s)", flush=True)
        self._stop.set()
    
    # ---- Main message handler (provider-specific via hooks) ----
    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        evt = json.loads(raw)
        t = evt.get("type", "")
        
        # Audio delta
        if t in ("response.audio.delta", "response.output_audio.delta"):
            self._audio_chunks += 1
            if self._first_delta_t is None:
                self._first_delta_t = time.monotonic()
                self._speaking = True
            if not self._current_item_id:
                self._current_item_id = evt.get("item_id")
                self._current_content_index = evt.get("content_index", 0)
            try:
                pcm_b64 = evt["delta"]
                self._audio_bytes_in_response += (len(pcm_b64) * 3) // 4
                self.handle_audio_delta({"delta": pcm_b64})
            except Exception as e:
                print(f"[speaker] push failed: {e}", flush=True)
        
        # Speech detection
        elif t == "input_audio_buffer.speech_started":
            self._barge_in()
        elif t == "input_audio_buffer.speech_stopped":
            pass
        
        # Response lifecycle
        elif t == "response.created":
            self._reset_response_state()
            self._response_active = True
        
        # Transcript
        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta",
                   "response.output_text.delta"):
            self.handle_transcript_delta(evt)
        
        elif t == "conversation.item.input_audio_transcription.completed":
            self.handle_input_transcription(evt)
        
        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            print(flush=True)
        
        # Tool calls
        elif t == "response.output_item.done":
            item = evt.get("item") or {}
            if item.get("type") == "function_call":
                self._handle_tool_call({
                    "name": item.get("name"),
                    "call_id": item.get("call_id"),
                    "arguments": item.get("arguments") or "{}",
                })
        
        # Response done
        elif t == "response.done":
            self._handle_response_done(evt)
        
        # Errors
        elif t == "error":
            err = evt.get("error") or {}
            code = err.get("code")
            if code not in ("response_cancel_not_active", "conversation_already_has_active_response"):
                print(f"[ws] error: {err}", flush=True)
    
    def _handle_response_done(self, evt: dict) -> None:
        """Handle response.done event."""
        rsp = evt.get("response") or {}
        rsp_id = rsp.get("id")
        if rsp_id and rsp_id == self._last_done_response_id:
            return
        self._last_done_response_id = rsp_id
        
        usage = rsp.get("usage") or {}
        cost, br = self.compute_cost(usage)
        self._cost_total += cost
        self._turns += 1
        print(
            f"[bot] response.done audio_chunks={self._audio_chunks} "
            f"cost=${cost:.4f} cumul=${self._cost_total:.4f} (turn #{self._turns})",
            flush=True,
        )
        
        # Play deferred emotion
        if self._pending_emotion:
            from reachy_mini.motion.recorded_move import RecordedMoves
            emotions = EmotionPlayer(
                self.mini, RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            )
            emotions.play(self._pending_emotion, min_interval=0.0)
            self._pending_emotion = None
        
        needs_followup = self._needs_followup_response
        self._needs_followup_response = False
        self._reset_response_state()
        
        if needs_followup:
            self._send({"type": "response.create"})
    
    # ---- Tool handling (shared) ----
    def _run_move_async(self, fn) -> None:
        """Run a movement off the WS thread."""
        prev = self._move_thread
        
        def _wrapper() -> None:
            if prev is not None and prev.is_alive():
                prev.join(timeout=10.0)
            try:
                fn()
            except Exception as e:
                print(f"[move] failed: {e}", flush=True)
        
        t = threading.Thread(target=_wrapper, daemon=True)
        self._move_thread = t
        t.start()
    
    def _handle_tool_call(self, evt: dict) -> None:
        """Handle function call from the provider."""
        name = evt.get("name")
        call_id = evt.get("call_id")
        args_raw = evt.get("arguments") or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        print(f"[tool] {name}({args})", flush=True)
        
        if name == "play_emotion":
            emo = args.get("name", "")
            if self._speaking:
                self._pending_emotion = emo
                result = f"queued:{emo}"
            else:
                from reachy_mini.motion.recorded_move import RecordedMoves
                emotions = EmotionPlayer(
                    self.mini, RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                )
                played = emotions.play(emo, min_interval=0.0)
                result = f"played:{emo}" if played else f"skipped:{emo}"
        
        elif name == "look":
            direction = args.get("direction", "center")
            pose = LOOK_POSES.get(direction)
            if pose is None:
                result = f"unknown_direction:{direction}"
            else:
                self._run_move_async(lambda: self.mini.goto_target(pose, duration=0.6))
                result = f"looking:{direction}"
        
        elif name == "move_sequence":
            steps = args.get("steps") or []
            if not steps:
                result = "empty_sequence"
            else:
                self._run_move_async(lambda: self._play_sequence(steps))
                result = f"playing:{len(steps)}_steps"
        
        else:
            result = f"unknown_tool:{name}"
        
        self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"status": result}),
            },
        })
        self._needs_followup_response = True
    
    def _play_sequence(self, steps: list) -> None:
        """Execute a planned head choreography."""
        import numpy as np
        from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
        
        for step in steps:
            try:
                roll = max(-30.0, min(30.0, float(step.get("roll", 0.0))))
                pitch = max(-30.0, min(30.0, float(step.get("pitch", 0.0))))
                yaw = max(-60.0, min(60.0, float(step.get("yaw", 0.0))))
                x_mm = max(-30.0, min(30.0, float(step.get("x", 0.0))))
                y_mm = max(-30.0, min(30.0, float(step.get("y", 0.0))))
                z_mm = max(-30.0, min(30.0, float(step.get("z", 0.0))))
                duration = max(0.1, min(3.0, float(step.get("duration", 0.4))))
                pose = _make_head_pose(roll, pitch, yaw, x_mm, y_mm, z_mm)
                
                antennas = None
                al = step.get("antenna_left")
                ar = step.get("antenna_right")
                if al is not None or ar is not None:
                    al_rad = np.deg2rad(max(-90.0, min(90.0, float(al or 0.0))))
                    ar_rad = np.deg2rad(max(-90.0, min(90.0, float(ar or 0.0))))
                    antennas = [al_rad, ar_rad]
                
                self.mini.goto_target(pose, antennas=antennas, duration=duration)
            except Exception as e:
                print(f"[seq] step failed: {e}", flush=True)
                return
        
        # back to neutral
        try:
            self.mini.goto_target(
                INIT_HEAD_POSE,
                antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                duration=0.5,
            )
        except Exception as e:
            print(f"[seq] return to neutral failed: {e}", flush=True)
    
    # ---- Run method ----
    def run(self) -> None:
        """Main run loop."""
        import signal
        
        self._print_config()
        
        headers = [self.get_auth_header()]
        self._ws = websocket.WebSocketApp(
            self.get_ws_url(),
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        
        def _sigint(_sig, _frm):
            self._stop.set()
            try:
                self._ws.close()
            except Exception:
                pass
        
        signal.signal(signal.SIGINT, _sigint)
        
        try:
            self._ws.run_forever()
        finally:
            try:
                self.media.stop_recording()
            except Exception:
                pass
            try:
                self.media.stop_playing()
            except Exception:
                pass
                self.media.stop_playing()
            except Exception:
                pass
