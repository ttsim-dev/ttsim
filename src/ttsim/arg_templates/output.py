from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ttsim.interface_dag_elements.typing import NestedTargetDict


@dataclass(frozen=True)
class ABC:
    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class Name(ABC):
    name: str | tuple[str, ...]
    """A single output name. Could be a qualified name or a tree path."""


@dataclass(frozen=True)
class Names(ABC):
    names: Iterable[str | tuple[str, ...]] | NestedTargetDict
    """An iterable of output names. Could be qualified names or tree paths."""
