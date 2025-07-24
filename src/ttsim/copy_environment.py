"""Utility function for copying policy environments and other tree structures."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, TypeAlias, overload

import optree

if TYPE_CHECKING:
    from ttsim.typing.interface_dag_elements import (
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
    )

    SomeEnv: TypeAlias = (
        PolicyEnvironment
        | SpecEnvWithoutTreeLogicAndWithDerivedFunctions
        | SpecEnvWithProcessedParamsAndScalars
        | SpecEnvWithPartialledParamsAndScalars
    )


@overload
def copy_environment(env: PolicyEnvironment) -> PolicyEnvironment: ...


@overload
def copy_environment(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions: ...


@overload
def copy_environment(
    env: SpecEnvWithProcessedParamsAndScalars,
) -> SpecEnvWithProcessedParamsAndScalars: ...


@overload
def copy_environment(
    env: SpecEnvWithPartialledParamsAndScalars,
) -> SpecEnvWithPartialledParamsAndScalars: ...


def copy_environment(env: SomeEnv) -> SomeEnv:
    """Create a copy of a policy environment or other tree structure.

    This function creates a copy of nested tree structures that may contain objects
    that cannot be deep-copied due to unpickleable elements such as function objects.

    The function uses optree.tree_map with shallow copy to create independent copies
    of the tree structure while preserving references to functions and other objects
    that don't need to be copied.

    Parameters
    ----------
    env
        The environment to copy. Can be a PolicyEnvironment or any of the
        specialized environment types (SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithProcessedParamsAndScalars, SpecEnvWithPartialledParamsAndScalars).

    Returns
    -------
    A copy of *env*, which is a deep copy for all practical purposes.

    """
    return optree.tree_map(copy, env)
