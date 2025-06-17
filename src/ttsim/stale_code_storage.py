from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree

from ttsim.tt_dag_elements import (
    ColumnObject,
    ParamFunction,
    ParamObject,
    policy_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    NestedAnyTTSIMObject = Mapping[
        str,
        ColumnObject
        | ParamFunction
        | ParamObject
        | int
        | float
        | bool
        | BoolColumn
        | IntColumn
        | FloatColumn
        | DatetimeColumn
        | "NestedAnyTTSIMObject",
    ]
    NestedAny = Mapping[str, Any | "NestedAnyTTSIMObject"]
    """Tree mapping TTSIM paths to any type of TTSIM object."""


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
    converted: NestedAnyTTSIMObject = optree.tree_map(
        lambda leaf: _convert_to_policy_function_if_callable(leaf),
        tree,  # type: ignore[arg-type]
    )  # type: ignore[assignment]
    return converted


def _convert_to_policy_function_if_callable(
    obj: ColumnObject | ParamFunction | Callable[..., Any] | Any,
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


@interface_function()
def name_of_last_branch_element_is_not_the_functions_leaf_name(
    functions_tree: NestedColumnObjectsParamFunctions,
) -> None:
    """Raise error if a PolicyFunction does not have the same leaf name as the last
    branch element of the tree path.
    """

    for tree_path, function in dt.flatten_to_tree_paths(functions_tree).items():
        if tree_path[-1] != function.leaf_name:
            raise KeyError(
                f"""
                The name of the last branch element of the functions tree must be the
                same as the leaf name of the PolicyFunction. The tree path {tree_path}
                is not compatible with the PolicyFunction {function.leaf_name}.
                """
            )


@pytest.mark.parametrize(
    "functions_tree",
    [
        {"foo": policy_function(leaf_name="bar")(return_one)},
    ],
)
def test_fail_if_name_of_last_branch_element_is_not_the_functions_leaf_name(
    functions_tree: NestedColumnObjectsParamFunctions,
):
    with pytest.raises(KeyError):
        name_of_last_branch_element_is_not_the_functions_leaf_name(functions_tree)


def check_series_has_expected_type(
    series: pd.Series, internal_type: numpy.dtype, dnp: ModuleType
) -> bool:
    """Checks whether used series has already expected internal type.

    Currently not used, but might become useful again.

    Parameters
    ----------
    series: pandas.Series or pandas.DataFrame or dict of pandas.Series
        Data provided by the user.
    internal_type: TypeVar
        One of the types used by TTSIM.

    Returns
    -------
    Bool

    """
    if (
        (internal_type == float) & (is_float_dtype(series))
        or (internal_type == int) & (is_integer_dtype(series))
        or (internal_type == bool) & (is_bool_dtype(series))
        or (internal_type == dnp.datetime64) & (is_datetime64_any_dtype(series))
    ):
        out = True
    else:
        out = False

    return out
