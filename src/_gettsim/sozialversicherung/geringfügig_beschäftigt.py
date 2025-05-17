"""Marginally employed."""

from ttsim import RoundingSpec, policy_function


@policy_function()
def geringfügig_beschäftigt(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    minijob_grenze: float,
) -> bool:
    """Individual earns less than marginal employment threshold.

    Marginal employed pay no social insurance contributions.

    Legal reference: § 8 Abs. 1 Satz 1 and 2 SGB IV
    """
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        <= minijob_grenze
    )


@policy_function(
    end_date="1999-12-31",
    leaf_name="minijob_grenze",
    rounding_spec=RoundingSpec(
        base=1, direction="up", reference="§ 8 Abs. 1a Satz 2 SGB IV"
    ),
)
def minijob_grenze_unterscheidung_ost_west(
    wohnort_ost: bool, geringfügige_einkommen_params: dict
) -> float:
    """Minijob income threshold depending on place of living (East or West Germany).

    Until 1999, the threshold is different for East and West Germany.
    """
    west = geringfügige_einkommen_params["grenzen_m"]["minijob"]["west"]
    ost = geringfügige_einkommen_params["grenzen_m"]["minijob"]["ost"]
    return ost if wohnort_ost else west


@policy_function(
    start_date="2000-01-01",
    end_date="2022-09-30",
    leaf_name="minijob_grenze",
    rounding_spec=RoundingSpec(
        base=1, direction="up", reference="§ 8 Abs. 1a Satz 2 SGB IV"
    ),
)
def minijob_grenze_fixer_betrag(geringfügige_einkommen_params: dict) -> float:
    """Minijob income threshold depending on place of living.

    From 2000 onwards, the threshold is the same for all of Germany. Until September
    2022, the threshold is exogenously set.
    """
    return geringfügige_einkommen_params["grenzen_m"]["minijob"]


@policy_function(
    start_date="2022-10-01",
    leaf_name="minijob_grenze",
    rounding_spec=RoundingSpec(
        base=1, direction="up", reference="§ 8 Abs. 1a Satz 2 SGB IV"
    ),
)
def minijob_grenze_abgeleitet_von_mindestlohn(
    geringfügige_einkommen_params: dict,
) -> float:
    """Minijob income threshold since 10/2022. Since then, it is calculated endogenously
    from the statutory minimum wage.
    """
    return (
        geringfügige_einkommen_params["mindestlohn"]
        * geringfügige_einkommen_params["faktor_minijobformel_zähler"]
        / geringfügige_einkommen_params["faktor_minijobformel_nenner"]
    )
