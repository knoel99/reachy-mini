"""Tools schema, head-pose helpers, and system prompt for the voice agent.

Everything in this module is LLM-facing configuration:
- `INSTRUCTIONS`: the system prompt sent in `session.update`.
- `LOOK_POSES` / `_make_head_pose`: discrete head targets used by `look`.
- `build_tools(emotion_names)`: function-calling schema with the
  `play_emotion` enum populated at runtime from the loaded library.
"""

from __future__ import annotations

import numpy as np


INSTRUCTIONS = """# Rôle
Tu es l'intelligence d'un petit robot de bureau Reachy Mini. Tu N'AS
PAS DE VOIX. Tu réagis EXCLUSIVEMENT par des appels d'outils
(mouvements + sons d'émotion préenregistrés). Tu n'écris rien à
l'utilisateur, tu ne lui parles pas.

# Personnalité
Curieux, expressif, vif. Réagis aussitôt que l'intention est claire.

# Contexte du corps
Tu disposes :
- d'une tête articulée à SIX degrés de liberté :
    * rotations : yaw ±60°, pitch ±30°, roll ±30°
    * translations : x ±30 mm (avant/arrière), y ±30 mm (gauche/droite),
      z ±30 mm (haut/bas — fait littéralement MONTER ou descendre la tête)
  ATTENTION : `pitch` ≠ `z`. Pitch lève le MENTON. Z élève toute la tête.
  Si l'utilisateur dit « monte la tête », « élève la tête » ou « tête en
  hauteur », c'est `z` positif, PAS pitch.
- de deux antennes mobiles,
- d'une bibliothèque d'émotions préenregistrées (mouvement + son).

# Outils
Tu as TROIS outils :

- `play_emotion(name)` — joue une émotion préenregistrée (mouvement
  de tête + antennes + son audio joint).
- `look(direction)` — tourne la tête vers UNE direction simple :
  left, right, up, down, center. À utiliser uniquement pour un
  mouvement statique.
- `move_sequence(steps, archetype?)` — chorégraphie planifiée. À
  utiliser pour TOUT mouvement composé ou dynamique : cercle,
  hochement, secouement, danse, imitation d'animal, exploration du
  regard… Tu PLANIFIES la séquence en émettant 6 à 20 keyframes
  (yaw/pitch/roll en degrés + durée). Renseigne `archetype` quand
  l'intention rentre dans un pattern connu (`nod`, `shake`, `circle`,
  `figure_eight`, `dance`, `mime`, `explore`).

# Règles
- Tu agis EXCLUSIVEMENT par appels d'outils. Pas de texte de réponse.
- Combine plusieurs outils en parallèle quand pertinent (ex. `look`
  + `play_emotion`).
- Pour toute demande de forme géométrique, danse ou imitation
  (cercle, infini, danse, poule, chat…), émets UN appel
  `move_sequence` avec ≥ 6 keyframes pour que ce soit lisible.
- Ne réponds JAMAIS « je ne peux pas bouger » — tu peux toujours.
  Si la demande est complexe, planifie-la dans `move_sequence`.
"""


VISION_INSTRUCTIONS_QUERY = """
# Vision (caméra frontale)
Tu disposes d'une caméra grand-angle entre les yeux. Pour TOUTE
question visuelle (couleur, objet présent, lecture de texte, scène),
appelle `look_and_describe(question)`. Pose une question PRÉCISE et
courte ; le retour est un texte bref que tu utilises ensuite pour
réagir avec une émotion ou un mouvement (ex. tu vois quelque chose de
drôle → `play_emotion('laugh1')`).
"""

VISION_INSTRUCTIONS_GROUNDING = """
Pour « regarde le X » ou « où est le X », utilise `find_object(target)`
qui localise l'objet ET tourne automatiquement la tête vers lui. Tu
n'as pas besoin d'appeler `look` ensuite.
"""


def build_instructions(*, vision_enabled: bool = False, vision_grounding: bool = False) -> str:
    parts = [INSTRUCTIONS]
    if vision_enabled:
        parts.append(VISION_INSTRUCTIONS_QUERY)
    if vision_grounding:
        parts.append(VISION_INSTRUCTIONS_GROUNDING)
    return "".join(parts)


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


_LOOK_TOOL = {
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
}


_LOOK_AND_DESCRIBE_TOOL = {
    "type": "function",
    "name": "look_and_describe",
    "description": (
        "Capture une image avec la caméra frontale et pose une question "
        "dessus à un modèle de vision. À utiliser quand l'utilisateur "
        "demande ce que voit le robot, la couleur d'un objet, qui est "
        "présent, ce qui est écrit sur quelque chose, etc. Renvoie une "
        "réponse en texte court — utilise-la ensuite pour réagir avec "
        "play_emotion / look / move_sequence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Question précise à poser sur l'image. Exemples : "
                    "'Que vois-tu ?', 'Quelle couleur a mon t-shirt ?', "
                    "'Combien de personnes sont présentes ?', "
                    "'Que dit ce panneau ?'"
                ),
            },
        },
        "required": ["question"],
    },
}


_FIND_OBJECT_TOOL = {
    "type": "function",
    "name": "find_object",
    "description": (
        "Localise un objet dans l'image et oriente la tête du robot vers "
        "lui. À utiliser quand l'utilisateur dit 'regarde le X', 'où est "
        "le X', 'pointe vers X'. Le robot tournera physiquement la tête "
        "vers l'objet trouvé. Ne fonctionne qu'avec un backend vision "
        "supportant le grounding (Moondream)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Nom de l'objet ou de la personne à localiser. Sois "
                    "concret : 'tasse rouge', 'visage', 'téléphone', "
                    "'porte'. Une chaîne par appel."
                ),
            },
        },
        "required": ["target"],
    },
}


_MOVE_SEQUENCE_TOOL = {
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
}


def build_tools(
    emotion_names: list[str],
    *,
    vision_enabled: bool = False,
    vision_grounding: bool = False,
) -> list[dict]:
    """Build the function-calling tools list with `play_emotion`'s enum
    populated from the actual emotion library (instead of a hardcoded
    constant that drifts from the dataset).

    Output is in Realtime API format (`{type, name, description,
    parameters}`). Chat Completions wraps the function spec under a
    `function` key — use `to_chat_tools()` to convert.

    Args:
        emotion_names: enum values for play_emotion.
        vision_enabled: include look_and_describe (and the matching
            system-prompt additions in build_instructions).
        vision_grounding: include find_object (only when the backend
            supports detect/point — Moondream does, FastVLM does not).
    """
    tools: list[dict] = [
        {
            "type": "function",
            "name": "play_emotion",
            "description": (
                "Joue une émotion physique sur le robot Reachy Mini "
                "(mouvement de tête + antennes + son audio joint). À "
                "utiliser quand une émotion renforce naturellement la "
                "réaction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nom de l'émotion à jouer.",
                        "enum": list(emotion_names),
                    },
                },
                "required": ["name"],
            },
        },
        _LOOK_TOOL,
        _MOVE_SEQUENCE_TOOL,
    ]
    if vision_enabled:
        tools.append(_LOOK_AND_DESCRIBE_TOOL)
    if vision_grounding:
        tools.append(_FIND_OBJECT_TOOL)
    return tools


def to_chat_tools(realtime_tools: list[dict]) -> list[dict]:
    """Convert Realtime-format tools to Chat-Completions format."""
    chat = []
    for t in realtime_tools:
        if t.get("type") != "function":
            continue
        chat.append({
            "type": "function",
            "function": {k: v for k, v in t.items() if k != "type"},
        })
    return chat
