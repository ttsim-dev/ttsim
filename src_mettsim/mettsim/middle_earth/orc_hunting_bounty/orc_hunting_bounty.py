from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from ttsim.tt import (
    UNSET_UNIT,
    TTSIMUnit,
    param_function,
    policy_function,
    policy_input,
)

if TYPE_CHECKING:
    from ttsim.tt import ConsecutiveIntLookupTableParamValue
    from ttsim.typing import RawParamValue


@dataclass(frozen=True)
class BountyPerLargeOrc:
    noble_hunter: Annotated[float, TTSIMUnit.CURRENCY]
    peasant_hunter: Annotated[float, TTSIMUnit.CURRENCY]


@dataclass(frozen=True)
class BountyPerOrc:
    small_orc: Annotated[int, TTSIMUnit.CURRENCY]
    large_orc: BountyPerLargeOrc
    minimum_hunter_age: Annotated[int, TTSIMUnit.YEARS]


@param_function(unit=UNSET_UNIT)
def bounty_per_orc(raw_bounties_per_orc: RawParamValue) -> BountyPerOrc:
    return BountyPerOrc(
        small_orc=raw_bounties_per_orc["small_orc"],
        large_orc=BountyPerLargeOrc(
            noble_hunter=raw_bounties_per_orc["large_orc"]["noble_hunter"],
            peasant_hunter=raw_bounties_per_orc["large_orc"]["peasant_hunter"],
        ),
        minimum_hunter_age=raw_bounties_per_orc["minimum_hunter_age"],
    )


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def small_orcs_hunted() -> int:
    """The number of small orcs hunted."""


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def large_orcs_hunted() -> int:
    """The number of large orcs hunted."""


@policy_function(unit=TTSIMUnit.CURRENCY)
def amount(
    amount_without_topup: float,
    bounty_topup_by_age: ConsecutiveIntLookupTableParamValue,
    age: int,
) -> float:
    """Orc-hunting bounty."""
    return amount_without_topup * bounty_topup_by_age.look_up(age)


@policy_function(unit=TTSIMUnit.CURRENCY)
def amount_without_topup(
    age: int,
    small_orcs_hunted: int,
    large_orcs_hunted: int,
    parent_is_noble: bool,
    bounty_per_orc: BountyPerOrc,
) -> float:
    """Orc-hunting bounty without topup; nothing for a hunter below the minimum age."""
    bounty_small_orcs = bounty_per_orc.small_orc * small_orcs_hunted
    if parent_is_noble:
        bounty_large_orcs = bounty_per_orc.large_orc.noble_hunter * large_orcs_hunted
    else:
        bounty_large_orcs = bounty_per_orc.large_orc.peasant_hunter * large_orcs_hunted
    if age < bounty_per_orc.minimum_hunter_age:
        out = 0.0
    else:
        out = bounty_small_orcs + bounty_large_orcs
    return out
