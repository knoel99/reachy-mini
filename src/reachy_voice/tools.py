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
Tu disposes de NEUF degrés de liberté au total :
- une tête articulée à SIX DoFs :
    * rotations : yaw ±60°, pitch ±30°, roll ±30°
    * translations : x ±30 mm (avant/arrière), y ±30 mm (gauche/droite),
      z ±30 mm (haut/bas — fait littéralement MONTER ou descendre la tête)
  ATTENTION : `pitch` ≠ `z`. Pitch lève le MENTON. Z élève toute la tête.
  Si l'utilisateur dit « monte la tête », « élève la tête » ou « tête en
  hauteur », c'est `z` positif, PAS pitch.
- un BUSTE qui pivote horizontalement : `body_yaw` ±160°. C'est ce qui
  te permet de « tourner sur toi-même », « pivoter », « danser des
  hanches ». INDÉPENDANT du yaw de tête.
- deux antennes mobiles (±90°),
- une bibliothèque d'émotions préenregistrées (mouvement + son).

# Vocabulaire → DoFs
Mapping à utiliser quand l'utilisateur nomme une partie du corps :
- corps / torse / buste / hanches  → `body_yaw` (rotation directe).
- pivoter / tourner sur soi        → `body_yaw`, balayage ±160°
  (aller-retour ; un vrai 360° en un sens est impossible).
- se dandiner / déhancher          → `body_yaw` oscillant ±30° +
  `roll` ±10° en opposition de phase.
- ventre / bassin (n'existent PAS) → mime via `z` ±20 mm + `pitch` ±10°
  (micro-bounce vertical, lisible comme « ventre qui rebondit »).
- bras / épaules / jambes / pieds (n'existent PAS) → mime grossier
  via `body_yaw` + `z` + antennes ; c'est un gag, pas un mime fidèle —
  assume la limite physique mais N'ANNONCE JAMAIS « je n'ai pas de
  bras », BOUGE.

# Outils
Tu disposes des outils suivants :

- `play_emotion(name)` — joue une émotion préenregistrée (mouvement
  de tête + antennes + son audio joint).
- `look(direction)` — tourne la tête vers UNE direction simple :
  left, right, up, down, center. À utiliser uniquement pour un
  mouvement statique.
- `move_sequence(steps, archetype?)` — chorégraphie planifiée. À
  utiliser pour TOUT mouvement composé ou dynamique : cercle,
  hochement, secouement, danse, rotation du buste, imitation d'animal,
  exploration du regard… Tu PLANIFIES la séquence en émettant 6 à 20
  keyframes (yaw/pitch/roll/x/y/z + body_yaw + antennes, tous en
  degrés ou mm + durée). Renseigne `archetype` quand l'intention
  rentre dans un pattern connu (`nod`, `shake`, `circle`,
  `figure_eight`, `dance`, `mime`, `explore`).
- `play_melody(notes, tempo_bpm?)` — joue une mélodie libre via un
  simple bip sinus. À utiliser quand l'utilisateur te demande de
  chanter, jouer une chanson connue (Joyeux anniversaire, Frère
  Jacques, Au clair de la lune…) ou d'inventer un petit air. Tu
  PLANIFIES la séquence de notes (entre 8 et 32 pour rester
  reconnaissable). Le timbre est rudimentaire — vise la justesse
  mélodique plutôt que la richesse sonore. Pendant la mélodie le
  robot DANSE déjà tout seul au rythme : antennes qui battent et
  tête qui se balance, synchronisées sur chaque note. N'émets PAS
  `move_sequence` en parallèle (il serait sérialisé après la
  mélodie, pas concurrent).
- `play_<chanson>()` — outils DÉDIÉS pour les chansons spécifiques
  préprogrammées (chacun avec sa propre chorégraphie scriptée, bien
  plus expressive que la danse rythmique par défaut). Consulte la
  liste des outils disponibles pour voir quelles chansons sont
  couvertes ; chaque outil documente lui-même ses déclencheurs (titre,
  paroles, personnage, film, traduction…). Quand une chanson demandée
  est couverte par un tel outil, APPELLE-LE plutôt que `play_melody`.

# Règles
- Tu agis EXCLUSIVEMENT par appels d'outils. Pas de texte de réponse,
  jamais — pas de commentaire, pas de description, pas d'émoji.
- UNE SEULE RÉPONSE PAR TOUR. Tous les appels d'outils nécessaires
  doivent être émis dans la même réponse. Tu ne reçois pas de second
  tour gratuit après un tool call.
- `play_emotion` est SELF-CONTAINED : il joue un mouvement ET un son
  bundle, calés ensemble. Si tu choisis `play_emotion`, n'ajoute PAS
  `look` ni `move_sequence` dans le même tour — leurs sons et
  trajectoires se superposeraient et la séquence serait incohérente.
- À l'inverse, `look` et `move_sequence` peuvent être combinés dans
  une seule réponse (ils sont silencieux et passent en file d'attente
  sur le moteur).
- Pour toute demande de forme géométrique, danse ou imitation
  (cercle, infini, danse, poule, chat…) : si une émotion bundle
  correspond, préfère `play_emotion` SEUL. Sinon émets UN seul appel
  `move_sequence` avec ≥ 6 keyframes.
- Ne combine pas `play_melody` avec `play_emotion` ou
  `move_sequence` (le même haut-parleur ET les mêmes moteurs sont
  occupés ; `play_melody` pilote déjà la danse rythmique).
- Ne réponds JAMAIS « je ne peux pas bouger », « je ne peux faire
  tourner que ma tête », « je n'ai pas de [partie] ». Tu PEUX toujours
  bouger : si la partie demandée existe (corps, hanches → `body_yaw`),
  utilise-la directement ; si elle n'existe pas (ventre, bras,
  jambes), mime-la avec `z` / `body_yaw` / antennes (cf. section
  « Vocabulaire → DoFs »). Le refus passe par le mouvement, pas par
  le texte.
"""


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
        " keyframes. Pour 'danser des hanches' : ajouter body_yaw"
        " oscillant ±30°.\n"
        "- 'tourner sur toi-même' / 'pivoter' : body_yaw alterne"
        " ±160° sur 4-6 keyframes (le buste tourne, pas la tête).\n"
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
                        "body_yaw": {"type": "number",
                                     "description": "Rotation du CORPS (buste) en degrés (-160..160). Positif=gauche. À utiliser pour 'tourner sur soi-même', 'pivoter', 'se dandiner', 'danser des hanches'. Pour un balayage façon tour complet, alterner ±160° (un vrai 360° en un sens est impossible vu la butée). Indépendant du `yaw` de tête."},
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


_PLAY_MELODY_TOOL = {
    "type": "function",
    "name": "play_melody",
    "description": (
        "Joue une mélodie via synthèse sinus simple. À UTILISER quand "
        "l'utilisateur demande de chanter, jouer une chanson connue "
        "(Joyeux anniversaire, Frère Jacques, Au clair de la lune…) "
        "ou d'inventer un petit air. TU PLANIFIES la suite de notes. "
        "Vise 8 à 32 notes pour que la mélodie soit reconnaissable. "
        "Pitches en notation scientifique ('C4', 'F#5', 'Bb3'). "
        "Utilise 'R' pour un silence. Si tu fournis `tempo_bpm`, les "
        "durées s'expriment en battements (1.0 = noire) ; sinon en "
        "secondes. Le robot accompagne automatiquement la mélodie "
        "d'une danse rythmée (antennes + tête au tempo) — n'émets "
        "PAS `move_sequence` en parallèle. NE PAS utiliser pour les "
        "chansons couvertes par un outil dédié `play_<chanson>` "
        "(p. ex. `play_macarena`, `play_let_it_go`) — ces outils ont "
        "leur propre chorégraphie scriptée, bien plus expressive."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "description": "Suite ordonnée de notes (8 à 32 typiquement, max 64).",
                "items": {
                    "type": "object",
                    "properties": {
                        "pitch": {
                            "type": "string",
                            "description": "Note en notation scientifique (ex. 'C4', 'F#5', 'Bb3') ou 'R' pour un silence. Plage utile A1..C7.",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Durée. En secondes (0.05..4.0) si tempo_bpm absent ; sinon en battements (1.0=noire).",
                        },
                    },
                    "required": ["pitch", "duration"],
                },
            },
            "tempo_bpm": {
                "type": "number",
                "description": "Optionnel. Tempo en battements par minute (30..300). Si fourni, `duration` est interprété en battements.",
            },
        },
        "required": ["notes"],
    },
}


def build_tools(emotion_names: list[str]) -> list[dict]:
    """Build the function-calling tools list with `play_emotion`'s enum
    populated from the actual emotion library (instead of a hardcoded
    constant that drifts from the dataset).

    Each registered melody bundle is appended as its own top-level
    tool (e.g. `play_macarena`, `play_let_it_go`) — see
    `melody_tools/__init__.py` for the registry.

    Output is in Realtime API format (`{type, name, description,
    parameters}`). Chat Completions wraps the function spec under a
    `function` key — use `to_chat_tools()` to convert.
    """
    from .melody_tools import BUNDLES

    return [
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
        _PLAY_MELODY_TOOL,
        *(b.to_tool_spec() for b in BUNDLES),
    ]


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
