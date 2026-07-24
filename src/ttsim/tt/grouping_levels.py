"""Registration of grouping levels as pint base dimensions (GEP 10)."""

from __future__ import annotations

from collections.abc import Iterable

import pint

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    CompositeUnit,
    _grouping_level_unit_name,
    _unit_builder_levels,
)


def register_unit_builder_levels(names: Iterable[str]) -> None:
    """Give the fluent builder a ``per_<level>`` attribute for each level.

    The level vocabulary is open — each policy environment brings its own levels
    via its ``*_id`` columns — so the builder cannot hard-wire the level step the
    way it does the closed area/period steps. Each package calls this at import,
    before its declarations run, so they can spell ``.PER_HH``-style steps. This
    is spelling vocabulary for the fluent DSL only; the level *dimensions* are
    registered per build from the environment's ``*_id`` columns
    (:func:`register_grouping_levels`). Idempotent.
    """
    names = list(names)
    fail_if_grouping_level_names_are_invalid(names=names)
    for name in names:
        if name in _unit_builder_levels:
            continue
        _unit_builder_levels.add(name)
        setattr(
            CompositeUnit,
            f"PER_{name.upper()}",
            property(lambda self, level=name: self.PER_LEVEL(level)),
        )


def fail_if_grouping_level_names_are_invalid(names: Iterable[str]) -> None:
    """Reject a grouping-level name the builder cannot own.

    A level claims the builder step ``PER_<NAME>`` on :class:`CompositeUnit`,
    which is a process-global class shared by every system. Three kinds of
    names are therefore refused:

    - any name whose step is already one of the closed area/period steps, since
      a ``month`` level would turn ``PER_MONTH`` from a flow period into a
      grouping level for every declaration in the process;
    - any name that is not lower-case, because a level is registered verbatim but
      resolved lower-cased (:func:`ttsim.tt.units.resolve_compositional_unit`), so
      ``"HH"`` would register a level that ``.PER_HH`` cannot resolve.

    Levels are derived from the policy environment's ``*_id`` columns — one per
    group column — so a ``person_id`` or ``month_id`` column reaches this check.

    Raises:
        UnitDefinitionError: If any name is refused.
    """
    for name in names:
        if name != name.lower():
            raise UnitDefinitionError(
                f"Grouping level {name!r} must be lower-case: a level is registered "
                f"under the name given but resolved lower-cased, so {name!r} would "
                f"register a level that `.PER_{name.upper()}` cannot resolve. Spell it "
                f"{name.lower()!r} (GEP 10)."
            )
        step = f"PER_{name.upper()}"
        if name not in _unit_builder_levels and hasattr(CompositeUnit, step):
            raise UnitDefinitionError(
                f"Grouping level {name!r} would claim the builder step {step}, which "
                f"is already a unit denominator. Rename the group so its level does "
                f"not collide with an area or period (GEP 10)."
            )


def register_grouping_levels(names: Iterable[str], registry: pint.UnitRegistry) -> None:
    """Register grouping levels as base dimensions in a policy system's registry.

    Args:
        names: The grouping-level names to register (e.g. ``["hh", "bg"]``). registry:
        The policy system's registry to define the dimensions in.
    """
    names = list(names)
    define_grouping_level_dimensions(names=names, registry=registry)
    for name in names:
        _ALLOWED_UNIT_TOKENS.add(_grouping_level_unit_name(name))
    # Packages that use the builder at import time call
    # `register_unit_builder_levels` directly, before their declarations run.
    register_unit_builder_levels(names)


def define_grouping_level_dimensions(
    names: Iterable[str], registry: pint.UnitRegistry
) -> None:
    """Define each grouping level's base dimension in ``registry``."""
    names = list(names)
    fail_if_grouping_level_names_are_invalid(names=names)
    for name in names:
        unit_name = _grouping_level_unit_name(name)
        if unit_name not in registry:
            registry.define(f"{unit_name} = [{unit_name}]")
