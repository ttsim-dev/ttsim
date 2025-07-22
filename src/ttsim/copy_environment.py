"""Utility function for copying policy environments and other tree structures."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, TypeAlias, overload

import optree

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
    )

    TreeLike: TypeAlias = (
        PolicyEnvironment
        | SpecEnvWithoutTreeLogicAndWithDerivedFunctions
        | SpecEnvWithProcessedParamsAndScalars
        | SpecEnvWithPartialledParamsAndScalars
    )


@overload
def copy_environment(tree: PolicyEnvironment) -> PolicyEnvironment: ...


@overload
def copy_environment(
    tree: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions: ...


@overload
def copy_environment(
    tree: SpecEnvWithProcessedParamsAndScalars,
) -> SpecEnvWithProcessedParamsAndScalars: ...


@overload
def copy_environment(
    tree: SpecEnvWithPartialledParamsAndScalars,
) -> SpecEnvWithPartialledParamsAndScalars: ...


def copy_environment(tree: TreeLike) -> TreeLike:
    """Create a copy of a policy environment or other tree structure.

    This function creates a copy of nested tree structures that may contain objects
    that cannot be deep-copied due to unpickleable elements such as function objects.

    The function uses optree.tree_map with shallow copy to create independent copies
    of the tree structure while preserving references to functions and other objects
    that don't need to be copied.

    Parameters
    ----------
    tree
        The tree structure to copy. Can be a PolicyEnvironment or any of the
        specialized environment types (SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithProcessedParamsAndScalars, SpecEnvWithPartialledParamsAndScalars).

    Returns
    -------
    The same type as the input tree
        A copy of the input tree where:
        - The nested structure is recreated (independent dictionaries at each level)
        - Leaf objects (like ScalarParam instances) are shallow-copied
        - Function references and other uncopyable objects are preserved as-is

    Examples
    --------
    Copy a policy environment and modify parameters independently:

    >>> from gettsim import main, MainTarget
    >>> from ttsim import copy_environment
    >>> from ttsim.tt_dag_elements.param_objects import ScalarParam
    >>> policy_env = main(date_str="2025-01-01", main_target=MainTarget.policy_environment)
    >>> copied_env = copy_environment(policy_env)
    >>> # Modify copy without affecting original
    >>> param_path = ["sozialversicherung", "rente", "beitrag", "beitragssatz"]
    >>> copied_env[param_path[0]][param_path[1]][param_path[2]][param_path[3]] = ScalarParam(value=0.3)

    Notes
    -----
    This function is preferred over copy.deepcopy for policy environments because:

    1. copy.deepcopy fails on policy environments containing function objects that
       cannot be pickled (error: "cannot pickle 'module' object" or similar)
    2. For policy environments, shallow copying of parameter objects is usually
       sufficient since the parameter values are typically immutable
    3. This approach is more robust and doesn't require modifying existing classes

    The function works with any optree-compatible tree structure, not just policy
    environments.
    """
    return optree.tree_map(copy, tree)
