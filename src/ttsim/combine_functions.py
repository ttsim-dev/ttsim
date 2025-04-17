from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.shared import (
    format_errors_and_warnings,
    format_list_linewise,
)
from ttsim.ttsim_objects import (
    TTSIMFunction,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        QualNameDataDict,
        QualNameTargetList,
        QualNameTTSIMFunctionDict,
        QualNameTTSIMObjectDict,
    )


def combine_policy_functions_and_derived_functions(
    ttsim_objects: QualNameTTSIMObjectDict,
    targets: QualNameTargetList,
    data: QualNameDataDict,
    groupings: tuple[str, ...],
) -> QualNameTTSIMFunctionDict:
    """Add derived functions to the qualified functions dict.

    Derived functions are time converted functions and aggregation functions (aggregate
    by p_id or by group).

    Checks that all targets have a corresponding function in the functions tree or can
    be taken from the data.

    Parameters
    ----------
    functions
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    targets
        The list of targets with qualified names.
    data
        Dict with qualified data names as keys and pandas Series as values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.

    """
    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        ttsim_objects=ttsim_objects,
        data=data,
        groupings=groupings,
    )
    current_functions = {
        **{qn: f for qn, f in ttsim_objects.items() if isinstance(f, TTSIMFunction)},
        **time_conversion_functions,
    }
    # Create aggregation functions by group.
    aggregate_by_group_functions = create_agg_by_group_functions(
        ttsim_functions_with_time_conversions=current_functions,
        data=data,
        targets=targets,
        groupings=groupings,
    )
    current_functions = {**aggregate_by_group_functions, **current_functions}

    _fail_if_targets_not_in_functions(functions=current_functions, targets=targets)

    return current_functions


def _fail_if_targets_not_in_functions(
    functions: QualNameTTSIMFunctionDict, targets: QualNameTargetList
) -> None:
    """Fail if some target is not among functions.

    Parameters
    ----------
    functions
        Dictionary containing functions to build the DAG.
    targets
        The targets which should be computed. They limit the DAG in the way that only
        ancestors of these nodes need to be considered.

    Raises
    ------
    ValueError
        Raised if any member of `targets` is not among functions.

    """
    targets_not_in_functions_tree = [
        str(dt.tree_path_from_qual_name(n)) for n in targets if n not in functions
    ]
    if targets_not_in_functions_tree:
        formatted = format_list_linewise(targets_not_in_functions_tree)
        msg = format_errors_and_warnings(
            f"The following targets have no corresponding function:\n\n{formatted}"
        )
        raise ValueError(msg)
