"""Reproducer for Python 3.14 annotation extraction bug.

This reproduces the exact failure where annotation extraction fails in Python 3.14
when functions decorated with @policy_function go through vectorization and rounding,
then are passed to concatenate_functions with set_annotations=True.

The bug manifests as:
AnnotationMismatchError: function <function_name> has the argument type annotation
'<param_name>: no_annotation_found', but function <param_name> has return type:
BoolColumn.
"""

from __future__ import annotations

from ttsim.tt import RoundingSpec, policy_function


@policy_function()
def child_in_household_for_bug_reproducer(p_id: int) -> bool:
    """A function that returns a boolean column (for Python 3.14 bug reproducer).

    Returns True for p_id 0 and 2, False for p_id 1 to match expected test output.
    """
    return p_id != 1


@policy_function(
    leaf_name="net_income_parents_m",
    rounding_spec=RoundingSpec(
        base=1, direction="down", reference="Python 3.14 bug reproducer"
    ),
)
def net_income_parents_m(
    wealth: float,  # Use an existing input
    # This annotation extraction fails in Python 3.14
    payroll_tax__child_tax_credit__child_in_household_for_bug_reproducer: bool,
) -> float:
    """A function that takes a bool parameter and has rounding."""
    if payroll_tax__child_tax_credit__child_in_household_for_bug_reproducer:
        out = wealth * 0.8
    else:
        out = wealth
    return out
