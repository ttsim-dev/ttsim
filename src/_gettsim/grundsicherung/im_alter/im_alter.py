"""Grundsicherung im Alter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _gettsim.arbeitslosengeld_2.regelbedarf import RegelsatzNachRegelbedarfsstufen

from ttsim import policy_function


@policy_function()
def betrag_m_eg(
    arbeitslosengeld_2__regelbedarf_m_bg: float,
    mehrbedarf_schwerbehinderung_g_m_eg: float,
    kindergeld__betrag_m_eg: float,
    unterhalt__tatsächlich_erhaltener_betrag_m_eg: float,
    unterhaltsvorschuss__betrag_m_eg: float,
    einkommen_m_eg: float,
    erwachsene_alle_rentenbezieher_hh: bool,
    vermögen_eg: float,
    vermögensfreibetrag_eg: float,
    arbeitslosengeld_2__anzahl_kinder_eg: int,
    arbeitslosengeld_2__anzahl_personen_eg: int,
) -> float:
    """Calculate Grundsicherung im Alter on household level.

    # ToDo: There is no check for Wohngeld included as Wohngeld is
    # ToDo: currently not implemented for retirees.

    """

    # TODO(@ChristianZimpelmann): Treatment of Bedarfsgemeinschaften with both retirees
    # and unemployed job seekers probably incorrect
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/703

    # TODO(@MImmesberger): Check which variable is the correct Regelbedarf in place of
    # `arbeitslosengeld_2__regelbedarf_m_bg`
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/702

    # TODO (@MImmesberger): Remove `arbeitslosengeld_2__anzahl_kinder_eg ==
    # arbeitslosengeld_2__anzahl_personen_eg` condition once
    # `erwachsene_alle_rentenbezieher_hh`` is replaced by a more accurate
    # variable.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/696

    # Wealth check
    # Only pay Grundsicherung im Alter if all adults are retired (see docstring)
    if (
        (vermögen_eg >= vermögensfreibetrag_eg)
        or (not erwachsene_alle_rentenbezieher_hh)
        or (
            arbeitslosengeld_2__anzahl_kinder_eg
            == arbeitslosengeld_2__anzahl_personen_eg
        )
    ):
        out = 0.0
    else:
        # Subtract income
        out = (
            arbeitslosengeld_2__regelbedarf_m_bg
            + mehrbedarf_schwerbehinderung_g_m_eg
            - einkommen_m_eg
            - unterhalt__tatsächlich_erhaltener_betrag_m_eg
            - unterhaltsvorschuss__betrag_m_eg
            - kindergeld__betrag_m_eg
        )

    return max(out, 0.0)


@policy_function(start_date="2011-01-01")
def mehrbedarf_schwerbehinderung_g_m(
    schwerbehindert_grad_g: bool,
    arbeitslosengeld_2__anzahl_erwachsene_eg: int,
    mehrbedarf_bei_schwerbehinderungsgrad_g: float,
    arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
) -> float:
    """Calculate additional allowance for individuals with disabled person's pass G."""

    mehrbedarf_single = (
        arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen.rbs_1.regelsatz
    ) * mehrbedarf_bei_schwerbehinderungsgrad_g
    mehrbedarf_in_couple = (
        arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen.rbs_2.regelsatz
    ) * mehrbedarf_bei_schwerbehinderungsgrad_g

    if (schwerbehindert_grad_g) and (arbeitslosengeld_2__anzahl_erwachsene_eg == 1):
        out = mehrbedarf_single
    elif (schwerbehindert_grad_g) and (arbeitslosengeld_2__anzahl_erwachsene_eg > 1):
        out = mehrbedarf_in_couple
    else:
        out = 0.0

    return out


@policy_function(start_date="2005-01-01")
def vermögensfreibetrag_eg(
    arbeitslosengeld_2__anzahl_erwachsene_fg: int,
    arbeitslosengeld_2__anzahl_kinder_fg: int,
    parameter_vermögensfreibetrag: dict[str, float],
) -> float:
    """Calculate wealth not considered for Grundsicherung im Alter on household level."""
    return (
        parameter_vermögensfreibetrag["erwachsene"]
        * arbeitslosengeld_2__anzahl_erwachsene_fg
        + parameter_vermögensfreibetrag["kinder"] * arbeitslosengeld_2__anzahl_kinder_fg
    )
