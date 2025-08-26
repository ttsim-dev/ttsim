from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim.tt import param_function, policy_function, policy_input

if TYPE_CHECKING:
    from ttsim.tt import ConsecutiveIntLookupTableParamValue, RawParam


@dataclass(frozen=True)
class BountyPerLargeOrc:
    noble_hunter: float
    peasant_hunter: float


@dataclass(frozen=True)
class BountyPerOrc:
    small_orc: int
    large_orc: BountyPerLargeOrc


@param_function()
def bounty_per_orc(raw_bounties_per_orc: RawParam) -> BountyPerOrc:
    return BountyPerOrc(
        small_orc=raw_bounties_per_orc["small_orc"],
        large_orc=BountyPerLargeOrc(
            noble_hunter=raw_bounties_per_orc["large_orc"]["noble_hunter"],
            peasant_hunter=raw_bounties_per_orc["large_orc"]["peasant_hunter"],
        ),
    )


@policy_input()
def small_orcs_hunted() -> int:
    """The number of small orcs hunted."""


@policy_input()
def large_orcs_hunted() -> int:
    """The number of large orcs hunted."""


@policy_function()
def amount(
    amount_without_topup: float,
    bounty_topup_by_age: ConsecutiveIntLookupTableParamValue,
    age: int,
) -> float:
    """Orc-hunting bounty."""
    return amount_without_topup * bounty_topup_by_age.look_up(age)


@policy_function()
def amount_without_topup(
    small_orcs_hunted: int,
    large_orcs_hunted: int,
    parent_is_noble: bool,
    bounty_per_orc: BountyPerOrc,
) -> float:
    """Orc-hunting bounty without topup."""
    bounty_small_orcs = bounty_per_orc.small_orc * small_orcs_hunted
    if parent_is_noble:
        bounty_large_orcs = bounty_per_orc.large_orc.noble_hunter * large_orcs_hunted
    else:
        bounty_large_orcs = bounty_per_orc.large_orc.peasant_hunter * large_orcs_hunted
    return bounty_small_orcs + bounty_large_orcs
