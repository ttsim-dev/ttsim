"""Orc Hunting Bounty"""

from ttsim import DictTTSIMParam, policy_function


@policy_function(vectorization_strategy="vectorize")
def amount(
    small_orcs_hunted: int,
    large_orcs_hunted: int,
    parent_is_noble: bool,
    orc_hunting_bounty: DictTTSIMParam,
) -> float:
    """Orc-hunting bounty."""
    bounty_small_orcs = orc_hunting_bounty["small_orc"] * small_orcs_hunted
    if parent_is_noble:
        bounty_large_orcs = (
            orc_hunting_bounty["large_orc"]["hunter_noble"] * large_orcs_hunted
        )
    else:
        bounty_large_orcs = (
            orc_hunting_bounty["large_orc"]["hunter_not_noble"] * large_orcs_hunted
        )
    return bounty_small_orcs + bounty_large_orcs
