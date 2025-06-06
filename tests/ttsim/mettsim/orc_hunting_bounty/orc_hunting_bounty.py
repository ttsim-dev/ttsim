from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim.tt_dag_elements import param_function, policy_function

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import RawParam


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


@policy_function(vectorization_strategy="vectorize")
def amount(
    small_orcs_hunted: int,
    large_orcs_hunted: int,
    parent_is_noble: bool,
    bounty_per_orc: BountyPerOrc,
) -> float:
    """Orc-hunting bounty."""
    bounty_small_orcs = bounty_per_orc.small_orc * small_orcs_hunted
    if parent_is_noble:
        bounty_large_orcs = bounty_per_orc.large_orc.noble_hunter * large_orcs_hunted
    else:
        bounty_large_orcs = bounty_per_orc.large_orc.peasant_hunter * large_orcs_hunted
    return bounty_small_orcs + bounty_large_orcs
