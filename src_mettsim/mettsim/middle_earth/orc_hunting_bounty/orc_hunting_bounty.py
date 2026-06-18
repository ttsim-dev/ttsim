from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim.tt import Unit, param_function, policy_function, policy_input

if TYPE_CHECKING:
    from ttsim.tt import ConsecutiveIntLookupTableParamValue
    from ttsim.typing import RawParamValue


@dataclass(frozen=True)
class BountyPerLargeOrc:
    noble_hunter: float
    peasant_hunter: float


@dataclass(frozen=True)
class BountyPerOrc:
    small_orc: int
    large_orc: BountyPerLargeOrc


# Returns a structured dataclass, not a scalar; the dry-run cannot unit-check it,
# so opt out of body inference (GEP 10).
@param_function(unit=Unit.CURRENCY, verify_units=False)
def bounty_per_orc(raw_bounties_per_orc: RawParamValue) -> BountyPerOrc:
    return BountyPerOrc(
        small_orc=raw_bounties_per_orc["small_orc"],
        large_orc=BountyPerLargeOrc(
            noble_hunter=raw_bounties_per_orc["large_orc"]["noble_hunter"],
            peasant_hunter=raw_bounties_per_orc["large_orc"]["peasant_hunter"],
        ),
    )


@policy_input(unit=Unit.DIMENSIONLESS)
def small_orcs_hunted() -> int:
    """The number of small orcs hunted."""


@policy_input(unit=Unit.DIMENSIONLESS)
def large_orcs_hunted() -> int:
    """The number of large orcs hunted."""


@policy_function(unit=Unit.CURRENCY, verify_units=False)
def amount(
    amount_without_topup: float,
    bounty_topup_by_age: ConsecutiveIntLookupTableParamValue,
    age: int,
) -> float:
    """Orc-hunting bounty."""
    return amount_without_topup * bounty_topup_by_age.look_up(age)


# `bounty_per_orc` is a structured parameter object, not a plain quantity, so the
# dry-run cannot evaluate this body; opt out of unit inference (GEP 10).
@policy_function(unit=Unit.CURRENCY, verify_units=False)
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
