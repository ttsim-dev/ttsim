"""References to the law of policy objects (GEP 11).

TTSIM knows where references are stored, but not how to interpret them. Policy
functions carry their references in a ``Reference:`` section of the docstring,
parameters in the ``reference`` key of each date entry, and rounding rules in the
``reference`` argument of a :class:`~ttsim.tt.rounding.RoundingSpec`.

The interpretation of a reference is left to a parser that the country package
supplies via ``main(legal_references_parser=...)``.
"""

from __future__ import annotations

import re
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Marker of the section holding the references in the docstring of a policy function.
REFERENCE_SECTION_MARKER = "Reference:"

_NO_REFERENCE_RE = re.compile(r"^none\s*\((?P<reason>.*\S.*)\)$", re.DOTALL)


@dataclass(frozen=True)
class NoLegalReference:
    """A policy object that does not have a counterpart in the law.

    Written as ``none (<reason>)`` in place of a reference.
    """

    reason: str


#: The references of a policy object: either the reference strings or the reason why
#: there is no reference.
LegalReferences = tuple[str, ...] | NoLegalReference


@runtime_checkable
class LegalReferenceParser(Protocol):
    """Interpretation of references, supplied by the country package.

    The parsed objects are opaque to TTSIM. Their string representation should
    yield the reference in canonical form.
    """

    def parse_reference(self, reference: str) -> Hashable:
        """Parse a reference as written in the code (strict).

        Raises a ValueError if the reference is malformed.
        """
        ...

    def parse_query(self, query: str) -> Hashable:
        """Parse a reference as entered by a user (lenient).

        Raises a ValueError if the query cannot be interpreted.
        """
        ...

    def matches(self, query: Hashable, reference: Hashable) -> bool:
        """Whether a parsed reference matches a parsed query."""
        ...


def _from_single_string(text: str) -> LegalReferences:
    text = " ".join(text.split())
    no_reference = _NO_REFERENCE_RE.match(text)
    if no_reference:
        return NoLegalReference(reason=no_reference.group("reason").strip())
    return (text,)


def _reference_section(lines: list[str]) -> tuple[str, list[str]] | None:
    """The text after the marker and the lines that continue the section."""
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().startswith(REFERENCE_SECTION_MARKER)
        ),
        None,
    )
    if start is None:
        return None
    marker_line = lines[start]
    marker_indent = len(marker_line) - len(marker_line.lstrip())
    first = marker_line.strip()[len(REFERENCE_SECTION_MARKER) :].strip()
    section: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or len(line) - len(line.lstrip()) <= marker_indent:
            break
        section.append(line.strip())
    return first, section


def _list_items(section: list[str]) -> tuple[str, ...]:
    items: list[str] = []
    for line in section:
        if line.startswith("-"):
            items.append(line[1:].strip())
        elif items:
            items[-1] = f"{items[-1]} {line}"
        else:
            raise ValueError(
                "The items of a `Reference:` section must start with '-', got:\n\n"
                f"{line}"
            )
    if not items:
        raise ValueError("The `Reference:` section of the docstring is empty.")
    return tuple(" ".join(item.split()) for item in items)


def legal_references_from_docstring(docstring: str | None) -> LegalReferences | None:
    """Extract the ``Reference:`` section from a docstring.

    A single reference follows the colon on the same line. Several references are
    written as indented list items starting with ``-``. Lines that are indented more
    deeply than the line they follow continue that line.

    Returns None if the docstring does not contain a ``Reference:`` section.
    """
    if not docstring:
        return None
    found = _reference_section(docstring.expandtabs().splitlines())
    if found is None:
        return None
    first, section = found
    if first:
        return _from_single_string(" ".join([first, *section]))
    return _list_items(section)


def legal_references_from_spec(
    reference: str | list[str] | tuple[str, ...] | None,
) -> LegalReferences | None:
    """Normalize the ``reference`` of a parameter or a RoundingSpec.

    Returns None if no reference is given.
    """
    if reference is None:
        return None
    if isinstance(reference, str):
        return _from_single_string(reference)
    return tuple(" ".join(r.split()) for r in reference)
