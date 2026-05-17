"""Exception hierarchy for ttsim.

All ttsim-defined exceptions inherit from `TTSIMError`. Beartype runtime
type-check violations are re-raised as the appropriate subclass via
`violation_door_type` in `ttsim._beartype_conf`, so user-facing failures keep
a single, documented entry point in the hierarchy.

Per-component subclasses correspond to the user-facing boundaries decorated
with `@beartype(conf=<COMPONENT_CONF>)`: entry points, factory methods on
helper dataclasses, and the policy / input / param / aggregation /
group-creation decorator factories.

Two pre-existing exception types are hoisted into the hierarchy. They keep
their definition site (so existing imports keep working) but their base
class now points back into `TTSIMError`:

- `ConflictingActivePeriodsError` (defined in
  `ttsim.interface_dag_elements.fail_if`)
- `TranslateToVectorizableError` (defined in `ttsim.tt.vectorization`)

Both are re-exported from this module for discoverability.
"""


class TTSIMError(Exception):
    """Base class for all ttsim-defined exceptions."""


class EntryPointError(TTSIMError):
    """Raised when a call to `ttsim.main()` has invalid arguments."""


class InputDataError(TTSIMError):
    """Raised when an `InputData.*` factory receives invalid arguments."""


class TTTargetsError(TTSIMError):
    """Raised when a `TTTargets.*` factory receives invalid arguments."""


class PolicyFunctionDefinitionError(TTSIMError):
    """Raised when an `@policy_function` declaration is invalid.

    Includes annotation/contract violations such as scalar annotations on a
    `vectorization_strategy="not_required"` function, or column annotations
    on a function that ttsim must auto-vectorize.
    """


class PolicyInputDefinitionError(TTSIMError):
    """Raised when an `@policy_input` declaration is invalid."""


class ParamFunctionDefinitionError(TTSIMError):
    """Raised when an `@param_function` declaration is invalid."""


class AggregationDefinitionError(TTSIMError):
    """Raised when an aggregation function declaration is invalid.

    Covers both `@agg_by_group_function` and `@agg_by_p_id_function`.
    """


class GroupCreationDefinitionError(TTSIMError):
    """Raised when a `@group_creation_function` declaration is invalid."""


class RoundingSpecError(TTSIMError):
    """Raised when a `RoundingSpec` is constructed with invalid arguments."""


# Two legacy exceptions pre-date this hierarchy and keep their definition
# sites (`ttsim.interface_dag_elements.fail_if` and `ttsim.tt.vectorization`)
# to avoid breaking existing imports. Their base classes have been changed
# to `TTSIMError`, so they are caught by `except TTSIMError`. Import them
# directly from their defining modules — importing here would create an
# import cycle:
#
# - `from ttsim.interface_dag_elements.fail_if import ConflictingActivePeriodsError`
# - `from ttsim.tt.vectorization import TranslateToVectorizableError`


__all__ = [
    "AggregationDefinitionError",
    "EntryPointError",
    "GroupCreationDefinitionError",
    "InputDataError",
    "ParamFunctionDefinitionError",
    "PolicyFunctionDefinitionError",
    "PolicyInputDefinitionError",
    "RoundingSpecError",
    "TTSIMError",
    "TTTargetsError",
]
