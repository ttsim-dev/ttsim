"""Registration of grouping levels as pint base dimensions (GEP 10).

The registration *operations* for the per-build grouping levels (one per ``*_id``
group column). There is no ``person`` level: an individual quantity is bare,
carrying no grouping level. A level's *dimension* lives in a policy system's
registry, so :func:`register_grouping_levels` takes one; the fluent builder step
it also adds sits on :class:`CompositeUnit`, which is a plain value shared by
every system.
"""

from __future__ import annotations

from collections.abc import Iterable

import pint

from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    _INDIVIDUAL_LEVEL_NORMALIZED_AWAY,
    CompositeUnit,
    _grouping_level_unit_name,
    _unit_builder_levels,
)


def register_unit_builder_levels(names: Iterable[str]) -> None:
    """Give the fluent builder a ``per_<level>`` attribute for each level.

    The level vocabulary is open and discovered per build, so the builder cannot
    hard-wire the level step the way it does the closed area/period steps. Each
    package registers its levels at import, before its declarations run.
    Idempotent.

    The deprecated ``.PER_PERSON`` step is registered too, but there is no
    ``person`` grouping level (GEP 10): it normalizes to the bare unit, adding no
    level. New code drops the suffix.
    """
    for name in (_INDIVIDUAL_LEVEL_NORMALIZED_AWAY, *names):
        if name in _unit_builder_levels:
            continue
        _unit_builder_levels.add(name)
        setattr(
            CompositeUnit,
            f"PER_{name.upper()}",
            property(lambda self, level=name: self.PER_LEVEL(level)),
        )


def register_grouping_levels(names: Iterable[str], registry: pint.UnitRegistry) -> None:
    """Register grouping levels as base dimensions in a policy system's registry.

    Each grouping level — one per ``*_id`` group column (``hh``, ``bg``, ``fg``,
    …) — is its *own* pint base dimension with no conversion to any other: a
    household holds a variable number of persons, so the levels are not units of
    one shared dimension (the way ``month`` and ``year`` are units of ``[time]``)
    but distinct, non-interconvertible base dimensions. The level set is
    discovered per build from the policy environment's ``*_id`` columns. There is
    no ``person`` level (GEP 10): an individual quantity is bare, and a head count
    is a dimensionless ``1 / [group]``.

    Each level is defined under an internal :data:`_GROUPING_LEVEL_PREFIX`-prefixed
    pint name anchoring a fresh base dimension and added to the closed pint-token
    vocabulary. Whether a level is already known is asked of ``registry`` itself,
    so every system's registry gets its own dimension for the levels it uses.
    Re-registering an already-known level is a tolerated no-op.

    Args:
        names: The grouping-level names to register (e.g. ``["hh", "bg"]``).
        registry: The policy system's registry to define the dimensions in.
    """
    for name in names:
        unit_name = _grouping_level_unit_name(name)
        if unit_name not in registry:
            registry.define(f"{unit_name} = [{unit_name}]")
        _ALLOWED_UNIT_TOKENS.add(unit_name)
    # Packages that use the builder at import time call
    # `register_unit_builder_levels` directly, before their declarations run.
    register_unit_builder_levels(names)
