"""`BeartypeConf` instances for ttsim's perimeter and internal claws.

`INTERNAL_CONF` is the default conf for the `ttsim` package-wide claw
registered in `ttsim/__init__.py`. Violations under that claw surface as
beartype's own `BeartypeCallHintViolation`, marking them as internal ttsim
bugs rather than user error.

The remaining confs (`ENTRY_POINT_CONF`, `INPUT_DATA_CONF`,
`TT_TARGETS_CONF`, `POLICY_FUNCTION_CONF`, `POLICY_INPUT_CONF`,
`PARAM_FUNCTION_CONF`, `AGGREGATION_CONF`, `GROUP_CREATION_CONF`,
`ROUNDING_SPEC_CONF`) are used by explicit `@beartype(conf=...)` decorators
on user-facing constructors and entry points. They map type violations to
the relevant `ttsim.exceptions.*` class, preserving the documented
exception hierarchy at the user boundary. The decorators stack on top of
the package claw and take precedence at the call sites they cover.
"""

from beartype import BeartypeConf, BeartypeStrategy

from ttsim.exceptions import (
    AggregationDefinitionError,
    EntryPointError,
    GroupCreationDefinitionError,
    InputDataError,
    ParamFunctionDefinitionError,
    PolicyFunctionDefinitionError,
    PolicyInputDefinitionError,
    RoundingSpecError,
    TTSIMError,
    TTTargetsError,
)


def project_conf(error_class: type[TTSIMError]) -> BeartypeConf:
    """Build a `BeartypeConf` that re-raises violations as `error_class`.

    `On` strategy: full O(n) container validation so every bad entry in a
    mapping/sequence is reported, not just one sampled element. The
    decorated entry points are called rarely (construction, main),
    so per-call cost is invisible.

    `is_pep484_tower=True`: respect the PEP-484 numeric tower so `int`
    satisfies `float`-typed parameters (matches the implicit numeric
    conversion that Python and ruff's PYI041 both assume).

    `violation_door_type` and `violation_param_type` both point at
    `error_class` so the same exception surfaces regardless of whether the
    violation originated in a function parameter, a return value, or a
    manual `die_if_unbearable` call.
    """
    return BeartypeConf(
        is_color=False,
        is_pep484_tower=True,
        strategy=BeartypeStrategy.On,
        violation_door_type=error_class,
        violation_param_type=error_class,
        violation_return_type=error_class,
    )


# Default conf for the package-wide claw on `ttsim` registered in
# `ttsim/__init__.py`. A type violation in any internal helper surfaces as
# beartype's own `BeartypeCallHintViolation` rather than a project
# exception. User-facing constructors layer their own
# `@beartype(conf=...)` decorators on top to map violations to project
# exceptions; those decorators take precedence at the call sites they
# cover.
INTERNAL_CONF = BeartypeConf(
    is_color=False,
    is_pep484_tower=True,
    strategy=BeartypeStrategy.On,
)

ENTRY_POINT_CONF = project_conf(EntryPointError)
INPUT_DATA_CONF = project_conf(InputDataError)
TT_TARGETS_CONF = project_conf(TTTargetsError)
POLICY_FUNCTION_CONF = project_conf(PolicyFunctionDefinitionError)
POLICY_INPUT_CONF = project_conf(PolicyInputDefinitionError)
PARAM_FUNCTION_CONF = project_conf(ParamFunctionDefinitionError)
AGGREGATION_CONF = project_conf(AggregationDefinitionError)
GROUP_CREATION_CONF = project_conf(GroupCreationDefinitionError)
ROUNDING_SPEC_CONF = project_conf(RoundingSpecError)
