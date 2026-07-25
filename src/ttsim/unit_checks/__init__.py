"""Whole-environment unit checks.

Builds on the per-function engine in :mod:`ttsim.tt.units` to verify a fully
assembled policy environment on two counts:

- **Every active node declares a TTSIM unit.** Author declarations (``unit=`` /
  ``unit:``) cover inputs, functions, parameters, and param functions; derived
  nodes (aggregations, time-conversion variants, group ids) and the framework
  date nodes get theirs assigned here.
- **Each function body agrees with its declaration.** Every ``@policy_function``
  / ``@param_function`` body is unit-checked on representative quantities built
  from its producers' resolved units. A body the check cannot evaluate must opt
  out with ``verify_units=False``.

The two layers of :mod:`ttsim.tt.units` both appear here: a node's declared
*TTSIM unit* (:class:`~ttsim.tt.units.CompositeUnit`) is resolved once against the
environment's registry, and every check downstream compares the resulting *pint
units*.

The check runs in four stages, one module each:

- :mod:`ttsim.unit_checks.contracts` — what a node's type annotations promise;
- :mod:`ttsim.unit_checks.resolution` — declared TTSIM units resolved to pint
  units, per node kind;
- :mod:`ttsim.unit_checks.execution` — abstract interpretation of a body over
  those units;
- :mod:`ttsim.unit_checks.declarations` — the environment-wide checks, which
  this module re-exports.

The two interface-DAG nodes exposing the resolved units live in
:mod:`ttsim.interface_dag_elements.unit_checks`.
"""

from ttsim.unit_checks.declarations import (
    fail_if_environment_units_are_inconsistent,
    fail_if_environment_units_are_missing,
    fail_if_input_units_are_inconsistent,
    fail_if_not_all_leaves_are_unit_annotated_columns,
    flatten_unit_annotated_input_tree,
)
from ttsim.unit_checks.resolution import (
    node_is_boolean,
    resolve_environment_units,
)

__all__ = [
    "fail_if_environment_units_are_inconsistent",
    "fail_if_environment_units_are_missing",
    "fail_if_input_units_are_inconsistent",
    "fail_if_not_all_leaves_are_unit_annotated_columns",
    "flatten_unit_annotated_input_tree",
    "node_is_boolean",
    "resolve_environment_units",
]
