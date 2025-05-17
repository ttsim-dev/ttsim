"""Income relevant for public health insurance contributions."""

from __future__ import annotations

from ttsim import params_function, policy_function


@params_function(end_date="1989-12-31", leaf_name="parameter_beitragsbemessungsgrenze")
def parameter_beitragsbemessungsgrenze_einheitlich_vor_wiedervereinigung(
    parameter_beitragsbemessungsgrenze_einheitlich: float,
) -> float:
    return parameter_beitragsbemessungsgrenze_einheitlich


@params_function(
    start_date="1990-01-01",
    end_date="2000-12-31",
    leaf_name="parameter_beitragsbemessungsgrenze",
)
def parameter_beitragsbemessungsgrenze_ab_1990(
    parameter_beitragsbemessungsgrenze_mit_ost_west_unterschied: dict[str, float],
) -> dict[str, float]:
    return parameter_beitragsbemessungsgrenze_mit_ost_west_unterschied


@params_function(
    start_date="2001-01-01", leaf_name="parameter_beitragsbemessungsgrenze"
)
def parameter_beitragsbemessungsgrenze_einheitlich_ab_2001(
    parameter_beitragsbemessungsgrenze_einheitlich: float,
) -> float:
    return parameter_beitragsbemessungsgrenze_einheitlich


@params_function(
    end_date="1989-12-31", leaf_name="parameter_bezugsgröße_selbstständige"
)
def parameter_bezugsgröße_selbstständige_vor_wiedervereinigung(
    raw_parameter_bezugsgröße_selbstständige_einheitlich: float,
) -> float:
    return raw_parameter_bezugsgröße_selbstständige_einheitlich


@params_function(
    start_date="1990-01-01", leaf_name="parameter_bezugsgröße_selbstständige"
)
def parameter_bezugsgröße_selbstständige_mit_ost_west_unterschied(
    raw_parameter_bezugsgröße_selbstständige_mit_ost_west_unterschied: dict[str, float],
) -> dict[str, float]:
    return raw_parameter_bezugsgröße_selbstständige_mit_ost_west_unterschied


@policy_function()
def einkommen_m(
    einkommen_regulär_beschäftigt_m: float,
    sozialversicherung__regulär_beschäftigt: bool,
) -> float:
    """Wage subject to public health insurance contributions.

    This affects marginally employed persons and high wages for above the assessment
    ceiling.
    """
    if sozialversicherung__regulär_beschäftigt:
        out = einkommen_regulär_beschäftigt_m
    else:
        out = 0.0
    return out


@policy_function()
def einkommen_regulär_beschäftigt_m(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragsbemessungsgrenze_m: float,
) -> float:
    """Income subject to public health insurance contributions.

    This does not consider reduced contributions for Mini- and Midijobs. Relevant for
    the computation of payroll taxes.
    """

    return min(
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m,
        beitragsbemessungsgrenze_m,
    )


@policy_function(start_date="1990-01-01")
def bemessungsgrundlage_selbstständig_m(
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m: float,
    bezugsgröße_selbstständige_m: float,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
    privat_versichert: bool,
    beitragsbemessungsgrenze_m: float,
    mindestanteil_bezugsgröße_selbstständige: float,
) -> float:
    """Self-employed income which is subject to health insurance contributions.

    The value is bounded from below and from above. Only affects those self-employed who
    voluntarily contribute to the public health system.

    Reference: §240 SGB V Abs. 4
    """
    # Calculate if self employed insures via public health insurance.
    if einkommensteuer__einkünfte__ist_selbstständig and not privat_versichert:
        out = min(
            beitragsbemessungsgrenze_m,
            max(
                bezugsgröße_selbstständige_m * mindestanteil_bezugsgröße_selbstständige,
                einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m,
            ),
        )
    else:
        out = 0.0

    return out


@policy_function(end_date="1989-12-31", leaf_name="beitragsbemessungsgrenze_m")
def beitragsbemessungsgrenze_m_vor_wiedervereinigung(
    parameter_beitragsbemessungsgrenze: float,
) -> float:
    """Income threshold up to which health insurance payments apply."""
    return parameter_beitragsbemessungsgrenze


@policy_function(
    start_date="1990-01-01",
    end_date="2000-12-31",
    leaf_name="beitragsbemessungsgrenze_m",
)
def beitragsbemessungsgrenze_m_mit_ost_west_unterschied(
    wohnort_ost: bool,
    parameter_beitragsbemessungsgrenze: dict[str, float],
) -> float:
    """Income threshold up to which health insurance payments apply."""
    return (
        parameter_beitragsbemessungsgrenze["ost"]
        if wohnort_ost
        else parameter_beitragsbemessungsgrenze["west"]
    )


@policy_function(start_date="2001-01-01", leaf_name="beitragsbemessungsgrenze_m")
def beitragsbemessungsgrenze_m_ohne_ost_west_unterschied(
    parameter_beitragsbemessungsgrenze: float,
) -> float:
    """Income threshold up to which health insurance payments apply."""
    return parameter_beitragsbemessungsgrenze


@policy_function(end_date="1989-12-31", leaf_name="bezugsgröße_selbstständige_m")
def bezugsgröße_selbstständige_m_vor_wiedervereinigung(
    parameter_bezugsgröße_selbstständige: float,
) -> float:
    """Threshold for self employment income subject to health insurance.

    Selecting by place of living the income threshold for self employed up to which the
    rate of health insurance contributions apply.
    """
    return parameter_bezugsgröße_selbstständige


@policy_function(start_date="1990-01-01", leaf_name="bezugsgröße_selbstständige_m")
def bezugsgröße_selbstständige_m_mit_ost_west_unterschied(
    wohnort_ost: bool,
    parameter_bezugsgröße_selbstständige: dict[str, float],
) -> float:
    """Threshold for self employment income subject to health insurance.

    Selecting by place of living the income threshold for self employed up to which the
    rate of health insurance contributions apply.
    """
    return (
        parameter_bezugsgröße_selbstständige["ost"]
        if wohnort_ost
        else parameter_bezugsgröße_selbstständige["west"]
    )


@policy_function()
def bemessungsgrundlage_rente_m(
    sozialversicherung__rente__altersrente__betrag_m: float,
    sozialversicherung__rente__private_rente_betrag_m: float,
    beitragsbemessungsgrenze_m: float,
) -> float:
    """Pension income which is subject to health insurance contribution."""
    return min(
        sozialversicherung__rente__altersrente__betrag_m
        + sozialversicherung__rente__private_rente_betrag_m,
        beitragsbemessungsgrenze_m,
    )
