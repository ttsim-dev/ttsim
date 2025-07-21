"""Utility function for copying policy environments and other tree structures."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import optree

if TYPE_CHECKING:
    from typing import Any


def copy_environment(tree: Any) -> Any:  # noqa: ANN401
    """Create a copy of a policy environment or other tree structure.

    This function creates a copy of nested tree structures (like policy environments)
    that may contain objects that cannot be deep-copied due to unpickleable elements
    such as function objects.

    The function uses optree.tree_map with shallow copy to create independent copies
    of the tree structure while preserving references to functions and other objects
    that don't need to be copied.

    Parameters
    ----------
    tree : Any
        The tree structure to copy. Typically a nested dictionary representing
        a policy environment, but can be any tree-like structure supported by optree.

    Returns
    -------
    Any
        A copy of the input tree where:
        - The nested structure is recreated (independent dictionaries at each level)
        - Leaf objects (like ScalarParam instances) are shallow-copied
        - Function references and other uncopyable objects are preserved as-is

    Examples
    --------
    >>> from gettsim import main, MainTarget
    >>> from ttsim import copy_environment
    >>> from ttsim.tt_dag_elements.param_objects import ScalarParam
    >>>
    >>> # Load a policy environment
    >>> policy_env = main(
    ...     date_str="2025-01-01", main_target=MainTarget.policy_environment
    ... )
    >>>
    >>> # Create a copy
    >>> copied_env = copy_environment(policy_env)
    >>>
    >>> # Modify the copy without affecting the original
    >>> path = ["sozialversicherung", "rente", "beitrag", "beitragssatz"]
    >>> copied_env[path[0]][path[1]][path[2]][path[3]] = ScalarParam(value=0.3)
    >>>
    >>> # Verify independence
    >>> print("Copy:", copied_env[path[0]][path[1]][path[2]][path[3]].value)
    >>> print("Original:", policy_env[path[0]][path[1]][path[2]][path[3]].value)

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
