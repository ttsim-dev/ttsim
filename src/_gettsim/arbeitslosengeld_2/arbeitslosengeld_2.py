"""Arbeitslosengeld II (unemployment benefit II)."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_function


@policy_function(start_date="2005-01-01")
def betrag_m_bg(
    anspruchshöhe_m_bg: float,
    vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg: bool,
    vorrangprüfungen__kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: bool,
    vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: bool,
    volljährige_alle_rentenbezieher_hh: bool,
) -> float:
    """Calculate final monthly subsistence payment on household level.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    # TODO (@MImmesberger): No interaction between Wohngeld/ALG2 and Grundsicherung im
    # Alter (SGB XII) is implemented yet. We assume for now that households with only
    # retirees are eligible for Grundsicherung im Alter but not for ALG2/Wohngeld. All
    # other households are not eligible for SGB XII, but SGB II / Wohngeld. Once this is
    # resolved, remove the `volljährige_alle_rentenbezieher_hh` condition.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/703
    if (
        vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg
        or vorrangprüfungen__kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg
        or vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg
        or volljährige_alle_rentenbezieher_hh
    ):
        out = 0.0
    else:
        out = anspruchshöhe_m_bg

    return out


@policy_function(start_date="2005-01-01")
def anspruchshöhe_m_bg(
    regelbedarf_m_bg: float,
    anzurechnendes_einkommen_m_bg: float,
    vermögen_bg: float,
    vermögensfreibetrag_bg: float,
) -> float:
    """Calculate potential basic subsistence (after income deduction and wealth check).

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    # Check wealth exemption
    if vermögen_bg > vermögensfreibetrag_bg:
        out = 0.0
    else:
        # Deduct income from various sources
        out = max(
            0.0,
            regelbedarf_m_bg - anzurechnendes_einkommen_m_bg,
        )

    return out
