"""Registry of bundled-melody tools.

Each submodule exports a `BUNDLE = MelodyBundle(...)`. To add a new
song: drop a new module here (e.g. `frere_jacques.py`), import it
below, and append its `BUNDLE` to `BUNDLES`. The dispatcher in
`_actions.py` and the tool builder in `tools.py` read from this list
generically — no other code needs to change.
"""

from __future__ import annotations

from ._spec import MelodyBundle
from . import let_it_go, macarena


BUNDLES: list[MelodyBundle] = [
    macarena.BUNDLE,
    let_it_go.BUNDLE,
]

BUNDLES_BY_TOOL_NAME: dict[str, MelodyBundle] = {
    b.tool_name: b for b in BUNDLES
}


__all__ = ["MelodyBundle", "BUNDLES", "BUNDLES_BY_TOOL_NAME"]
