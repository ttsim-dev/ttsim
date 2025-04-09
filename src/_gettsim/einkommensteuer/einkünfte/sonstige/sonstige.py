"""Sonstige Einkünfte according to § 22 EStG."""

from ttsim import piecewise_polynomial, policy_function


@policy_function()
def betrag_m(
    sonstige_einkünfte_ohne_renten_m: float,
    renteneinkünfte_m: float,
) -> float:
    """Total sonstige Einkünfte."""
    return sonstige_einkünfte_ohne_renten_m + renteneinkünfte_m


@policy_function()
def renteneinkünfte_m(
    rente_ertragsanteil: float,
    sozialversicherung__rente__altersrente__betrag_m: float,
    sozialversicherung__rente__private_rente_betrag_m: float,
) -> float:
    """Pension income counting towards taxable income."""
    return rente_ertragsanteil * (
        sozialversicherung__rente__altersrente__betrag_m
        + sozialversicherung__rente__private_rente_betrag_m
    )


@policy_function()
def rente_ertragsanteil(
    sozialversicherung__rente__jahr_renteneintritt: int, eink_st_params: dict
) -> float:
    """Share of pensions subject to income taxation."""
    return piecewise_polynomial(
        x=sozialversicherung__rente__jahr_renteneintritt,
        thresholds=eink_st_params["rente_ertragsanteil"]["thresholds"],
        rates=eink_st_params["rente_ertragsanteil"]["rates"],
        intercepts_at_lower_thresholds=eink_st_params["rente_ertragsanteil"][
            "intercepts_at_lower_thresholds"
        ],
    )
