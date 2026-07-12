"""Registration of grouping levels as pint base dimensions (GEP 10).

The registration *operations* for the per-build grouping levels (the individual
``person`` leaf and one per ``*_id`` group column). The level state and the
dimension helpers a declaration is resolved against live in
:mod:`ttsim.tt.units`; this module reads and mutates that shared state.
"""

from __future__ import annotations

from collections.abc import Iterable

from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    PERSON_LEVEL,
    UNIT_REGISTRY,
    CompositeUnit,
    _grouping_level_unit_name,
    _registered_grouping_levels,
    _unit_builder_levels,
)


def register_unit_builder_levels(names: Iterable[str]) -> None:
    """Give the fluent builder a ``per_<level>`` attribute for each level.

    The level vocabulary is open and discovered per build, so the builder cannot
    hard-wire the level step the way it does the closed area/period steps. Each
    package registers its levels at import, before its declarations run. The
    person leaf is always registered. Idempotent.
    """
    for name in (PERSON_LEVEL, *names):
        if name in _unit_builder_levels:
            continue
        _unit_builder_levels.add(name)
        setattr(
            CompositeUnit,
            f"PER_{name.upper()}",
            property(lambda self, level=name: self.PER_LEVEL(level)),
        )


def register_grouping_levels(names: Iterable[str]) -> None:
    """Register grouping levels as base dimensions in the registry.

    Each grouping level — the individual ``person`` (the leaf, identified by
    ``p_id``) and one per ``*_id`` group column (``hh``, ``bg``, ``fg``, …) — is
    its *own* pint base dimension with no conversion to any other: a household
    holds a variable number of persons, so the levels are not units of one shared
    dimension (the way ``month`` and ``year`` are units of ``[time]``) but
    distinct, non-interconvertible base dimensions. The level set is discovered
    per build from the policy environment's ``*_id`` columns.

    ``person`` (the leaf level) is always registered: it doubles as the
    ``[person]`` *count* dimension, the conversion factor between levels that lets
    head counts and per-person amounts cancel.

    Each level is defined under an internal :data:`_GROUPING_LEVEL_PREFIX`-prefixed
    pint name anchoring a fresh base dimension and added to the closed pint-token
    vocabulary. Re-registering an already-known level is a tolerated no-op,
    mirroring :func:`register_currency`.

    Args:
        names: The grouping-level names to register (e.g. ``["hh", "bg"]``).
            ``person`` is added unconditionally.
    """
    for name in (PERSON_LEVEL, *names):
        if name in _registered_grouping_levels:
            continue
        unit_name = _grouping_level_unit_name(name)
        UNIT_REGISTRY.define(f"{unit_name} = [{unit_name}]")
        _ALLOWED_UNIT_TOKENS.add(unit_name)
        _registered_grouping_levels.add(name)
    # Packages that use the builder at import time call
    # `register_unit_builder_levels` directly, before their declarations run.
    register_unit_builder_levels(names)
