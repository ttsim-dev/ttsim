"""Marginally employed."""

from dataclasses import dataclass

from ttsim import RoundingSpec, params_function, policy_function


@dataclass(frozen=True)
class MinijobGrenzeUnterschiedOstWest:
    west: float
    ost: float


@params_function(end_date="1989-12-31", leaf_name="minijobgrenze")
def minijobgrenze_einheitlich_vor_wiedervereinigung(
    parameter_minijobgrenze_einheitlich: float,
) -> float:
    """Minijob income threshold"""
    return parameter_minijobgrenze_einheitlich


@params_function(
    start_date="1990-01-01",
    end_date="1999-12-31",
    leaf_name="minijobgrenze_nach_wohnort",
)
def minijobgrenze_ost_west_unterschied(
    parameter_minijobgrenze_ost_west_unterschied: dict[str, float],
) -> dict[str, float]:
    """Minijob income threshold"""
    return parameter_minijobgrenze_ost_west_unterschied


@params_function(
    start_date="2000-01-01", end_date="2022-09-30", leaf_name="minijobgrenze"
)
def minijobgrenze_einheitlich_ab_2000(
    parameter_minijobgrenze_einheitlich: float,
) -> float:
    """Minijob income threshold"""
    return parameter_minijobgrenze_einheitlich


@policy_function()
def geringfügig_beschäftigt(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    minijobgrenze: float,
) -> bool:
    """Individual earns less than marginal employment threshold.

    Marginal employed pay no social insurance contributions.

    Legal reference: § 8 Abs. 1 Satz 1 and 2 SGB IV
    """
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        <= minijobgrenze
    )


@policy_function(
    start_date="1990-01-01",
    end_date="1999-12-31",
    leaf_name="minijobgrenze",
    rounding_spec=RoundingSpec(
        base=1, direction="up", reference="§ 8 Abs. 1a Satz 2 SGB IV"
    ),
)
def minijobgrenze_unterscheidung_ost_west(
    wohnort_ost: bool, minijobgrenze_nach_wohnort: dict[str, float]
) -> float:
    """Minijob income threshold depending on place of living (East or West Germany).

    Until 1999, the threshold is different for East and West Germany.
    """
    return (
        minijobgrenze_nach_wohnort["ost"]
        if wohnort_ost
        else minijobgrenze_nach_wohnort["west"]
    )


@policy_function(
    start_date="2022-10-01",
    leaf_name="minijobgrenze",
    rounding_spec=RoundingSpec(
        base=1, direction="up", reference="§ 8 Abs. 1a Satz 2 SGB IV"
    ),
)
def minijobgrenze_abgeleitet_von_mindestlohn(
    mindestlohn: float,
    faktoren_minijobformel: dict[str, float],
) -> float:
    """Minijob income threshold since 10/2022. Since then, it is calculated endogenously
    from the statutory minimum wage.
    """
    return (
        mindestlohn
        * faktoren_minijobformel["zähler"]
        / faktoren_minijobformel["nenner"]
    )
