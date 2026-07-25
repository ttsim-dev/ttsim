"""The interface-DAG nodes exposing an environment's units.

Both nodes read the fully assembled policy environment: one resolves every
annotated node's declared TTSIM unit against the environment's registry, the
other hands out the declared tokens themselves. The machinery behind the
resolution — and the checks that consume it — lives in
:mod:`ttsim.unit_validation`.
"""

from __future__ import annotations

from typing import (
    Any,
)

import pint

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    UnitSystem,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_validation import resolve_environment_units


@interface_function()
def resolved_pint_units(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    labels__grouping_levels: OrderedQNames,
    unit_system: UnitSystem,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """The pint unit every annotated node's declared TTSIM unit resolves to."""
    return resolve_environment_units(
        env=specialized_environment__without_tree_logic_and_with_derived_functions,
        grouping_levels=labels__grouping_levels,
        unit_system=unit_system,
    )


@interface_function()
def declared_ttsim_units(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
) -> dict[str, CompositeUnit]:
    """Each node's declared TTSIM unit, by qname."""
    return {
        qname: token
        for qname, obj in specialized_environment__without_tree_logic_and_with_derived_functions.items()  # noqa: E501
        if isinstance((token := getattr(obj, "unit", UNSET_UNIT)), CompositeUnit)
    }
