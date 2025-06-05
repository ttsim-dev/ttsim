from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree

from ttsim.tt_dag_elements.column_objects_param_function import (
    ColumnObject,
    ParamFunction,
    policy_function,
)

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
        GenericCallable,
        NestedAny,
        NestedAnyTTSIMObject,
    )


def _convert_plain_functions_to_policy_functions(
    tree: NestedAny,
) -> NestedAnyTTSIMObject:
    """Convert all plain functions in a tree to PolicyFunctions.

    Convenience function if users do not want to apply decorators in modifications of
    the taxes and transfers system.

    Parameters
    ----------
    tree
        The tree of functions to convert.

    Returns
    -------
    converted_tree
        The converted tree.

    """
    converted = optree.tree_map(
        lambda leaf: _convert_to_policy_function_if_callable(leaf),
        tree,
    )
    return converted


def _convert_to_policy_function_if_callable(
    obj: ColumnObject | ParamFunction | GenericCallable | Any,
) -> ColumnObject:
    """Convert a Callable to a PolicyFunction if it is not already a ColumnObject or
    ParamFunction. If it is not a Callable, return it unchanged.

    Parameters
    ----------
    obj
        The object to convert.

    Returns
    -------
    converted_object
        The converted object.

    """
    if isinstance(obj, (ColumnObject, ParamFunction)) or not callable(obj):
        converted_object = obj
    else:
        converted_object = policy_function(leaf_name=obj.__name__)(obj)

    return converted_object
