from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NewType, TypeAlias

if TYPE_CHECKING:
    import datetime

    NestedLookupDict: TypeAlias = dict[int, float | int | bool | "NestedLookupDict"]
    OrigParamSpec = (
        # Header
        dict[str, str | None | dict[Literal["de", "en"], str | None]]
        |
        # Parameters at one point in time
        dict[
            datetime.date,
            dict[Literal["note", "reference"] | str | int, Any],  # noqa: PYI051
        ]
    )
    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
