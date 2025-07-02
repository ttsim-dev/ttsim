from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MainArg:
    def to_dict(self) -> dict[str, Any]:
        return self.__dict__
