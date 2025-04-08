"""Input columns."""

from ttsim import policy_input


@policy_input(start_date="2023-01-01")
def bezug_im_vorjahr() -> bool:
    """Received Arbeitslosengeld II / Bürgergeld in previous year."""
    return False


# TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt once
# Bedarfsgemeinschaften are fully endogenous
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/763
@policy_input(start_date="2023-01-01")
def eigenbedarf_gedeckt() -> bool:
    """Received Arbeitslosengeld II / Bürgergeld in previous year."""
    return False


@policy_input(start_date="2005-01-01")
def p_id_einstandspartner() -> int:
    """Identifier of Einstandspartner."""
    return -999999999
