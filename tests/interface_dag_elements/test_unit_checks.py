"""Tests for unit-related interface-DAG nodes."""

from tests.test_unit_system import TEST_UNIT_SYSTEM
from ttsim.interface_dag_elements.unit_checks import (
    declared_ttsim_units,
    resolved_pint_units,
)
from ttsim.tt import UNSET_UNIT, TTSIMUnit, policy_function, policy_input


@policy_input(unit=UNSET_UNIT)
def unannotated_income_y() -> float:
    """Represent an input without a unit declaration."""


@policy_function(unit=TTSIMUnit.DIMENSIONLESS)
def flag(unannotated_income_y: float) -> bool:
    """Return whether income is positive."""
    return unannotated_income_y > 0.0


def test_declared_ttsim_units_excludes_unset_declarations():
    units = declared_ttsim_units(
        specialized_environment__without_tree_logic_and_with_derived_functions={
            "unannotated_income_y": unannotated_income_y,
            "flag": flag,
        }
    )

    assert units == {"flag": TTSIMUnit.DIMENSIONLESS}


def test_resolved_pint_units_resolves_environment_declarations():
    units = resolved_pint_units(
        specialized_environment__without_tree_logic_and_with_derived_functions={
            "flag": flag,
        },
        labels__grouping_levels=(),
        unit_system=TEST_UNIT_SYSTEM,
    )

    assert units["flag"] == TEST_UNIT_SYSTEM.registry.dimensionless
