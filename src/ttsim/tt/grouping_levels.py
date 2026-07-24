"""Registration of grouping levels as pint base dimensions (GEP 10).

The registration *operations* for grouping levels. The level set comes
exclusively from the policy environment's ``*_id`` columns — one level per group
column — and is registered per build (via
:func:`ttsim.interface_dag_elements.unit_checks.resolve_environment_units`).
There is no ``person`` level: an individual quantity is bare, carrying no
grouping level. A level's *dimension* lives in a policy system's registry, so
:func:`register_grouping_levels` takes one; the fluent builder step it also
adds sits on :class:`CompositeUnit`, which is a plain value shared by every
system, so packages register it at import via
:func:`register_unit_builder_levels`.
"""

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

    - ``person``, because there is no individual grouping level (GEP 10) — an
      individual quantity is bare;
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
        if name.lower() == "person":
            raise UnitDefinitionError(
                f"{name!r} is not a grouping level: an individual quantity is bare, "
                f"carrying no level (GEP 10). Drop it from the grouping levels."
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

    Each grouping level — one per ``*_id`` group column (``hh``, ``bg``, ``fg``,
    …) — is its *own* pint base dimension with no conversion to any other: a
    household holds a variable number of persons, so the levels are not units of
    one shared dimension (the way ``month`` and ``year`` are units of ``[time]``)
    but distinct, non-interconvertible base dimensions. The level set is derived
    from the policy environment's ``*_id`` columns and registered here at build
    time. There is
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
    """Define each grouping level's base dimension in ``registry``.

    The registry-local half of :func:`register_grouping_levels`, and the only half
    that can reject a name — a caller that must not widen the process-global
    vocabulary until every name is known to be definable runs this first.
    Defining an already-known level is a tolerated no-op.
    """
    names = list(names)
    fail_if_grouping_level_names_are_invalid(names=names)
    for name in names:
        unit_name = _grouping_level_unit_name(name)
        if unit_name not in registry:
            registry.define(f"{unit_name} = [{unit_name}]")
