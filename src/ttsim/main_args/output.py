from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim.main_args import MainArg

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ttsim.interface_dag_elements.typing import NestedTargetDict


@dataclass(frozen=True)
class Name(MainArg):
    name: str | tuple[str, ...]
    """A single output name. Could be a qualified name or a tree path."""


@dataclass(frozen=True)
class Names(MainArg):
    names: Iterable[str | tuple[str, ...]] | NestedTargetDict
    """An iterable of output names. Could be qualified names or tree paths."""
