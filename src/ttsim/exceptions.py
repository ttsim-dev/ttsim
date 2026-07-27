"""Exception hierarchy for ttsim.

All ttsim-defined exceptions inherit from `TTSIMError`. Beartype runtime
type-check violations are re-raised as the appropriate subclass via
`violation_door_type` in `ttsim._beartype_conf`, so user-facing failures keep
a single, documented entry point in the hierarchy.

Per-component subclasses correspond to the user-facing boundaries decorated
with `@beartype(conf=<COMPONENT_CONF>)`: entry points, factory methods on
helper dataclasses, and the policy / input / param / aggregation /
group-creation decorator factories.

Two further exceptions subclass `TTSIMError` from their original definition
sites rather than this module, to avoid an import cycle:

- `ConflictingActivePeriodsError` — `ttsim.interface_dag_elements.fail_if`
- `TranslateToVectorizableError` — `ttsim.tt.vectorization`

Both are caught by `except TTSIMError`; import them from their defining
modules.

`ttsim._unit_inference._UnitCheckError` and its subclasses also subclass
`TTSIMError` at their own site, but are internal control-flow signals thrown and
caught within the unit check rather than part of this user-facing vocabulary.
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


class UnitDefinitionError(TTSIMError):
    """Raised when a `unit=` declaration or a `UnitSystem` definition is invalid.

    Covers an unparseable unit string, a unit involving a dimension outside the
    closed GEP-10 vocabulary, and inconsistent currency registration.
    """


class UnitConsistencyError(TTSIMError):
    """Raised when a declared unit disagrees with the inferred or producer unit.

    Covers the per-function body check (inferred output unit vs. declared unit)
    and the DAG edge-consistency check (producer unit vs. consumer expectation).
    """


# `ConflictingActivePeriodsError` and `TranslateToVectorizableError` subclass
# `TTSIMError` from their definition sites (`ttsim.interface_dag_elements.fail_if`
# and `ttsim.tt.vectorization`) — re-importing them here would create an
# import cycle, so import them directly from their defining modules.


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
    "UnitConsistencyError",
    "UnitDefinitionError",
]
