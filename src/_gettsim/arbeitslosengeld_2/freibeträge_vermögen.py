"""Wealth checks for Arbeitslosengeld II/Bürgergeld."""

from ttsim import policy_function


@policy_function(end_date="2022-12-31")
def grundfreibetrag_vermögen(
    familie__kind: bool,
    alter: int,
    geburtsjahr: int,
    maximaler_grundfreibetrag_vermögen: float,
    vermögensgrundfreibetrag_je_lebensjahr: dict[int, float],
) -> float:
    """Calculate wealth exemptions based on individuals age.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    threshold_years = list(vermögensgrundfreibetrag_je_lebensjahr.keys())
    if geburtsjahr <= threshold_years[0]:
        out = next(iter(vermögensgrundfreibetrag_je_lebensjahr.values())) * alter
    elif (geburtsjahr >= threshold_years[1]) and (not familie__kind):
        out = list(vermögensgrundfreibetrag_je_lebensjahr.values())[1] * alter
    else:
        out = 0.0

    return min(out, maximaler_grundfreibetrag_vermögen)


# TODO(@MImmesberger): Parameter should be defined as a piecewise_constant.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/911
@policy_function(end_date="2022-12-31")
def maximaler_grundfreibetrag_vermögen(
    geburtsjahr: int,
    familie__kind: bool,
    obergrenze_vermögensgrundfreibetrag: dict[int, float],
) -> float:
    """Calculate maximal wealth exemptions by year of birth.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    threshold_years = list(obergrenze_vermögensgrundfreibetrag.keys())
    obergrenzen = list(obergrenze_vermögensgrundfreibetrag.values())
    if familie__kind:
        out = 0.0
    else:
        if geburtsjahr < threshold_years[1]:
            out = obergrenzen[0]
        elif geburtsjahr < threshold_years[2]:
            out = obergrenzen[1]
        elif geburtsjahr < threshold_years[3]:
            out = obergrenzen[2]
        else:
            out = obergrenzen[3]

    return out


@policy_function(start_date="2023-01-01")
def freibetrag_vermögen_in_karenzzeit_bg(
    schonvermögen_bürgergeld: dict[str, float],
    anzahl_personen_bg: int,
) -> float:
    """Calculate wealth exemptions since 2023 during Karenzzeit. This variable is also
    reffered to as 'erhebliches Vermögen'.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    return (
        schonvermögen_bürgergeld["während_karenzzeit"]
        + (anzahl_personen_bg - 1) * schonvermögen_bürgergeld["normaler_satz"]
    )


@policy_function(end_date="2022-12-31", leaf_name="freibetrag_vermögen_bg")
def freibetrag_vermögen_bg_bis_2022(
    grundfreibetrag_vermögen_bg: float,
    anzahl_kinder_bis_17_bg: int,
    anzahl_personen_bg: int,
    vermögensfreibetrag_austattung: float,
    vermögensgrundfreibetrag_je_kind: float,
) -> float:
    """Calculate actual exemptions until 2022.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    out = (
        grundfreibetrag_vermögen_bg
        + anzahl_kinder_bis_17_bg * vermögensgrundfreibetrag_je_kind
        + anzahl_personen_bg * vermögensfreibetrag_austattung
    )
    return out


@policy_function(start_date="2023-01-01", leaf_name="freibetrag_vermögen_bg")
def freibetrag_vermögen_bg_ab_2023(
    anzahl_personen_bg: int,
    freibetrag_vermögen_in_karenzzeit_bg: float,
    arbeitslosengeld_2_bezug_im_vorjahr: bool,
    schonvermögen_bürgergeld: dict[str, float],
) -> float:
    """Calculate actual wealth exemptions since 2023.

    During the first year (Karenzzeit), the wealth exemption is substantially larger.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if arbeitslosengeld_2_bezug_im_vorjahr:
        out = anzahl_personen_bg * schonvermögen_bürgergeld["normaler_satz"]
    else:
        out = freibetrag_vermögen_in_karenzzeit_bg

    return out
