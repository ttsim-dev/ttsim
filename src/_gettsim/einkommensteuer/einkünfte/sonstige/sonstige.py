"""Sonstige Einkünfte according to § 22 EStG."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

from ttsim.tt_dag_elements import (
    PiecewisePolynomialParamValue,
    piecewise_polynomial,
    policy_function,
)


@policy_function()
def betrag_m(
    ohne_renten_m: float,
    renteneinkünfte_m: float,
) -> float:
    """Total sonstige Einkünfte."""
    return ohne_renten_m + renteneinkünfte_m


@policy_function()
def renteneinkünfte_m(
    ertragsanteil_der_rente: float,
    sozialversicherung__rente__altersrente__betrag_m: float,
    geförderte_private_renteneinnahmen_m: float,
    private_renteneinnahmen_m: float,
    betriebliche_renteneinnahmen_m: float,
) -> float:
    """Pension income counting towards taxable income.

    Reference: § 22 EStG
    """
    return (
        ertragsanteil_der_rente
        * (sozialversicherung__rente__altersrente__betrag_m + private_renteneinnahmen_m)
        + betriebliche_renteneinnahmen_m
        + geförderte_private_renteneinnahmen_m
    )


@policy_function()
def ertragsanteil_der_rente(
    sozialversicherung__rente__jahr_renteneintritt: int,
    parameter_ertragsanteil_der_rente: PiecewisePolynomialParamValue,
    xnp: ModuleType,
) -> float:
    """Share of pensions subject to income taxation."""
    return piecewise_polynomial(
        x=sozialversicherung__rente__jahr_renteneintritt,
        parameters=parameter_ertragsanteil_der_rente,
        xnp=xnp,
    )


@policy_function()
def private_und_betriebliche_renteneinnahmen_m(
    private_renteneinnahmen_m: float,
    geförderte_private_renteneinnahmen_m: float,
    betriebliche_renteneinnahmen_m: float,
) -> float:
    """Private and occupational pension income."""
    return (
        private_renteneinnahmen_m
        + geförderte_private_renteneinnahmen_m
        + betriebliche_renteneinnahmen_m
    )
