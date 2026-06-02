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

import os

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

# GEP 9: runtime type checking is on by default; `TTSIM_BEARTYPE_CLAW=0` opts
# out. Reading the env var here — in the single module that builds every conf
# — makes the switch authoritative. When off, the strategy of every conf falls
# back to beartype's no-op `O0`, which reduces `@beartype(conf=...)` to the
# identity decorator. So opting out disables *all* checking: the package claw
# (additionally gated in `ttsim/__init__.py`), every perimeter decorator, and
# the synthesized forwarder in `tt/type_resolution.py` — i.e. pre-GEP
# behaviour, not just the package claw.
RUNTIME_TYPE_CHECKING_ENABLED = os.environ.get("TTSIM_BEARTYPE_CLAW", "1") != "0"
_STRATEGY = (
    BeartypeStrategy.On if RUNTIME_TYPE_CHECKING_ENABLED else BeartypeStrategy.O0
)


def project_conf(error_class: type[TTSIMError]) -> BeartypeConf:
    """Build a `BeartypeConf` that re-raises violations as `error_class`.

    Strategy: `On` when runtime checking is enabled — full O(n) container
    validation so every bad entry in a mapping/sequence is reported, not
    just one sampled element; the decorated entry points are called rarely
    (construction, main), so per-call cost is invisible. When
    `TTSIM_BEARTYPE_CLAW=0` the strategy is beartype's no-op `O0`, reducing
    every `@beartype(conf=...)` decorator built here to the identity
    decorator.

    `is_pep484_tower=True`: respect the PEP-484 numeric tower so `int`
    satisfies `float`-typed parameters (matches the implicit numeric
    conversion that Python and ruff's PYI041 both assume).

    `violation_door_type`, `violation_param_type`, and
    `violation_return_type` all point at `error_class` so the same exception
    surfaces regardless of whether the violation originated in a manual
    `die_if_unbearable` call, a function parameter, or a return value.
    """
    return BeartypeConf(
        is_color=False,
        is_pep484_tower=True,
        strategy=_STRATEGY,
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
    strategy=_STRATEGY,
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
