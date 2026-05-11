"""`MelodyBundle`: a preset melody + 1:1 choreography + LLM tool metadata.

Each bundle is exposed to the LLM as its own top-level function call
named `play_{name}`. The bundle owns its full triggering vocabulary in
its `description` field, so the generic `play_melody` tool stays free
of song-specific keywords.

To add a new song: drop a module under `melody_tools/` that exports
`BUNDLE = MelodyBundle(...)`, then import it from
`melody_tools/__init__.py` so the registry picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MelodyBundle:
    name: str
    description: str
    bpm: float
    notes: list[dict] = field(default_factory=list)
    keyframes: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.notes) != len(self.keyframes):
            raise ValueError(
                f"bundle {self.name!r}: {len(self.notes)} notes vs "
                f"{len(self.keyframes)} keyframes (must be 1:1)"
            )

    @property
    def tool_name(self) -> str:
        return f"play_{self.name}"

    def to_tool_spec(self) -> dict:
        """OpenAI Realtime tool-spec dict for this bundle (no params)."""
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}},
        }
