from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _OptimizeState:
    initialized: bool = False
    cache: Optional[Any] = None
    profile: Optional[dict] = None


_state = _OptimizeState()