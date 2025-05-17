"""Income relevant for calculation of Kinderzuschlag."""

from ttsim import (
    AggType,
    RoundingSpec,
    agg_by_group_function,
    policy_function,
)


@agg_by_group_function(agg_type=AggType.SUM)
def arbeitslosengeld_2__anzahl_kinder_bg(
    kindergeld__anzahl_ansprüche: int, bg_id: int
) -> int:
    pass


@policy_function()
def bruttoeinkommen_eltern_m(
    arbeitslosengeld_2__bruttoeinkommen_m: float,
    kindergeld__grundsätzlich_anspruchsberechtigt: bool,
    familie__erwachsen: bool,
) -> float:
    """Calculate parental gross income for calculation of child benefit.

    This variable is used to check whether the minimum income threshold for child
    benefit is met.
    """
    # TODO(@MImmesberger): Redesign the conditions in this function: False for adults
    # who do not have Kindergeld claims.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/704
    if familie__erwachsen and (not kindergeld__grundsätzlich_anspruchsberechtigt):
        out = arbeitslosengeld_2__bruttoeinkommen_m
    else:
        out = 0.0

    return out


@policy_function(
    rounding_spec=RoundingSpec(base=10, direction="down", reference="§ 6a Abs. 4 BKGG"),
    leaf_name="nettoeinkommen_eltern_m",
    end_date="2019-06-30",
)
def nettoeinkommen_eltern_m_mit_grober_rundung(
    arbeitslosengeld_2__nettoeinkommen_nach_abzug_freibetrag_m: float,
    kindergeld__grundsätzlich_anspruchsberechtigt: bool,
    familie__erwachsen: bool,
) -> float:
    """Parental income (after deduction of taxes, social insurance contributions, and
    other deductions) for calculation of child benefit.
    """
    # TODO(@MImmesberger): Redesign the conditions in this function: False for adults
    # who do not have Kindergeld claims.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/704
    if familie__erwachsen and (not kindergeld__grundsätzlich_anspruchsberechtigt):
        out = arbeitslosengeld_2__nettoeinkommen_nach_abzug_freibetrag_m
    else:
        out = 0.0
    return out


@policy_function(
    rounding_spec=RoundingSpec(base=1, direction="down", reference="§ 11 Abs. 2 BKGG"),
    leaf_name="nettoeinkommen_eltern_m",
    start_date="2019-07-01",
)
def nettoeinkommen_eltern_m_mit_genauer_rundung(
    arbeitslosengeld_2__nettoeinkommen_nach_abzug_freibetrag_m: float,
    kindergeld__grundsätzlich_anspruchsberechtigt: bool,
    familie__erwachsen: bool,
) -> float:
    """Parental income (after deduction of taxes, social insurance contributions, and
    other deductions) for calculation of child benefit.
    """
    # TODO(@MImmesberger): Redesign the conditions in this function: False for adults
    # who do not have Kindergeld claims.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/704
    if familie__erwachsen and (not kindergeld__grundsätzlich_anspruchsberechtigt):
        out = arbeitslosengeld_2__nettoeinkommen_nach_abzug_freibetrag_m
    else:
        out = 0.0
    return out


@policy_function(end_date="2022-06-30", leaf_name="maximales_nettoeinkommen_m_bg")
def maximales_nettoeinkommen_m_bg_vor_06_2022(
    erwachsenenbedarf_m_bg: float,
    arbeitslosengeld_2__anzahl_kinder_bg: int,
    kinderzuschl_params: dict,
) -> float:
    """Calculate maximum income to be eligible for additional child benefit
    (Kinderzuschlag).

    There is a maximum income threshold, depending on the need, plus the potential kiz
    receipt (§6a (1) Nr. 3 BKGG).
    """
    return (
        erwachsenenbedarf_m_bg
        + kinderzuschl_params["maximum"] * arbeitslosengeld_2__anzahl_kinder_bg
    )


@policy_function(
    start_date="2022-07-01",
    end_date="2022-12-31",
    leaf_name="maximales_nettoeinkommen_m_bg",
)
def maximales_nettoeinkommen_m_bg_ab_06_2022_bis_12_2022(
    erwachsenenbedarf_m_bg: float,
    arbeitslosengeld_2__anzahl_kinder_bg: int,
    arbeitslosengeld_2__kindersofortzuschlag: float,
    kinderzuschl_params: dict,
) -> float:
    """Calculate maximum income to be eligible for additional child benefit
    (Kinderzuschlag).

    There is a maximum income threshold, depending on the need, plus the potential kiz
    receipt (§6a (1) Nr. 3 BKGG).
    """
    return (
        erwachsenenbedarf_m_bg
        + kinderzuschl_params["maximum"] * arbeitslosengeld_2__anzahl_kinder_bg
        + arbeitslosengeld_2__kindersofortzuschlag
        * arbeitslosengeld_2__anzahl_kinder_bg
    )


@policy_function(start_date="2023-01-01", leaf_name="maximales_nettoeinkommen_m_bg")
def maximales_nettoeinkommen_m_bg_ab_01_2023(
    erwachsenenbedarf_m_bg: float,
    arbeitslosengeld_2__anzahl_kinder_bg: int,
    kinderzuschl_params: dict,
) -> float:
    """Calculate maximum income to be eligible for additional child benefit
    (Kinderzuschlag).

    Kindersofortzuschlag is included in maximum Kinderzuschlag.

    There is a maximum income threshold, depending on the need, plus the potential kiz
    receipt (§6a (1) Nr. 3 BKGG).
    """
    return (
        erwachsenenbedarf_m_bg
        + kinderzuschl_params["maximum"] * arbeitslosengeld_2__anzahl_kinder_bg
    )


@policy_function()
def mindestbruttoeinkommen_m_bg(
    arbeitslosengeld_2__anzahl_kinder_bg: int,
    familie__alleinerziehend_bg: bool,
    kinderzuschl_params: dict,
) -> float:
    """Calculate minimal claim of child benefit (kinderzuschlag).

    Min income to be eligible for KIZ (different for singles and couples) (§6a (1) Nr. 2
    BKGG).
    """
    if arbeitslosengeld_2__anzahl_kinder_bg == 0:
        out = 0.0
    elif familie__alleinerziehend_bg:
        out = kinderzuschl_params["mindesteinkommen_alleinerziehende"]
    else:
        out = kinderzuschl_params["mindesteinkommen_paare"]

    return out


@policy_function()
def anzurechnendes_einkommen_eltern_m_bg(
    nettoeinkommen_eltern_m_bg: float,
    erwachsenenbedarf_m_bg: float,
    kinderzuschl_params: dict,
) -> float:
    """Calculate parental income subtracted from child benefit.

    (§6a (6) S. 3 BKGG)
    """
    out = kinderzuschl_params["entzugsrate_elterneinkommen"] * (
        nettoeinkommen_eltern_m_bg - erwachsenenbedarf_m_bg
    )

    return max(out, 0.0)


@policy_function(vectorization_strategy="loop")
def kosten_der_unterkunft_m_bg(
    wohnbedarf_anteil_eltern_bg: float,
    arbeitslosengeld_2__bruttokaltmiete_m_bg: float,
    arbeitslosengeld_2__heizkosten_m_bg: float,
) -> float:
    """Calculate costs of living eligible to claim.

    Unlike ALG2, there is no check on whether living costs are "appropriate".
    """
    warmmiete_m_bg = (
        arbeitslosengeld_2__bruttokaltmiete_m_bg + arbeitslosengeld_2__heizkosten_m_bg
    )

    return wohnbedarf_anteil_eltern_bg * warmmiete_m_bg


@policy_function(vectorization_strategy="loop")
def wohnbedarf_anteil_eltern_bg(
    arbeitslosengeld_2__anzahl_kinder_bg: int,
    arbeitslosengeld_2__anzahl_erwachsene_bg: int,
    kinderzuschl_params: dict,
) -> float:
    """Calculate living needs broken down to the parents. Defined as parents'
    subsistence level on housing, divided by sum of subsistence level from parents and
    children.

    Reference: § 6a Abs. 5 S. 3 BKGG
    """
    ex_min = kinderzuschl_params["existenzminimum"]

    # Up to 10 children are considered
    considered_children = min(arbeitslosengeld_2__anzahl_kinder_bg, 10)
    single_oder_paar = (
        "single" if arbeitslosengeld_2__anzahl_erwachsene_bg == 1 else "paare"
    )

    return (
        ex_min["kosten_der_unterkunft"][single_oder_paar]
        + ex_min["heizkosten"][single_oder_paar]
    ) / (
        ex_min["kosten_der_unterkunft"][single_oder_paar]
        + ex_min["heizkosten"][single_oder_paar]
        + (
            considered_children
            * (
                ex_min["kosten_der_unterkunft"]["kinder"]
                + ex_min["heizkosten"]["kinder"]
            )
        )
    )


@policy_function()
def erwachsenenbedarf_m_bg(
    arbeitslosengeld_2__regelsatz_m_bg: float, kosten_der_unterkunft_m_bg: float
) -> float:
    """Aggregate relevant income and rental costs."""
    return arbeitslosengeld_2__regelsatz_m_bg + kosten_der_unterkunft_m_bg


@agg_by_group_function(agg_type=AggType.SUM)
def kinderzuschlag_spec_target(kinderzuschlag_source_field: bool, bg_id: int) -> int:
    pass
