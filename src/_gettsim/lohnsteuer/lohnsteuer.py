"""Withholding tax on earnings (Lohnsteuer)."""

from __future__ import annotations

from ttsim import (
    PiecewisePolynomialParamValue,
    param_function,
    piecewise_polynomial,
    policy_function,
)
from ttsim.config import numpy_or_jax as np


def basis_für_klassen_5_6(
    einkommen_y: float, parameter_einkommensteuertarif: PiecewisePolynomialParamValue
) -> float:
    """Calculate base for Lohnsteuer for steuerklasse 5 and 6, by applying
    obtaining twice the difference between applying the factors 1.25 and 0.75
    to the lohnsteuer payment.

    §39 b Absatz 2 Satz 7 (part 1):

        Jahreslohnsteuer die sich aus dem Zweifachen des Unterschiedsbetrags zwischen
        dem Steuerbetrag für das Eineinviertelfache und dem Steuerbetrag für das
        Dreiviertelfache des zu versteuernden Jahresbetrags nach § 32a Absatz 1 ergibt;
        die Jahreslohnsteuer beträgt jedoch mindestens 14 Prozent des zu versteuernden
        Jahresbetrags.

    """

    return 2 * (
        piecewise_polynomial(einkommen_y * 1.25, parameter_einkommensteuertarif)
        - piecewise_polynomial(einkommen_y * 0.75, parameter_einkommensteuertarif)
    )


@param_function(start_date="2015-01-01")
def parameter_max_lohnsteuer_klasse_5_6(
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    einkommensgrenzwerte_steuerklassen_5_6: dict[int, float],
) -> PiecewisePolynomialParamValue:
    """Create Paramter Values for the piecewise polynomial that represents the maximum amount of Lohnsteuer
    that can be paid on incomes higher than the income thresholds for Steuerklasse 5 and 6.
    """
    lohnsteuer_bis_erste_grenze = basis_für_klassen_5_6(
        einkommensgrenzwerte_steuerklassen_5_6[1],
        einkommensteuer__parameter_einkommensteuertarif,
    )
    lohnsteuer_bis_zweite_grenze = basis_für_klassen_5_6(
        einkommensgrenzwerte_steuerklassen_5_6[2],
        einkommensteuer__parameter_einkommensteuertarif,
    )
    lohnsteuer_bis_dritte_grenze = basis_für_klassen_5_6(
        einkommensgrenzwerte_steuerklassen_5_6[3],
        einkommensteuer__parameter_einkommensteuertarif,
    )
    thresholds = np.asarray(
        [
            0,
            einkommensgrenzwerte_steuerklassen_5_6[1],
            einkommensgrenzwerte_steuerklassen_5_6[2],
            einkommensgrenzwerte_steuerklassen_5_6[3],
        ]
    )
    intercepts = np.asarray(
        [
            0,
            lohnsteuer_bis_erste_grenze,
            lohnsteuer_bis_zweite_grenze,
            lohnsteuer_bis_dritte_grenze,
        ]
    )
    rates = np.expand_dims(
        einkommensteuer__parameter_einkommensteuertarif.rates[0][[3, 3, 3, 4]], axis=0
    )
    parameter_max_lohnsteuer_klasse_5_6 = PiecewisePolynomialParamValue(
        thresholds=thresholds, intercepts=intercepts, rates=rates
    )

    return parameter_max_lohnsteuer_klasse_5_6


@policy_function(start_date="2015-01-01")
def basistarif(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
) -> float:
    """Lohnsteuer in the Basistarif."""
    return piecewise_polynomial(
        einkommen_y, einkommensteuer__parameter_einkommensteuertarif
    )


@policy_function(start_date="2015-01-01")
def splittingtarif(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
) -> float:
    """Lohnsteuer in the Splittingtarif."""
    return 2 * piecewise_polynomial(
        einkommen_y / 2, einkommensteuer__parameter_einkommensteuertarif
    )


@policy_function(start_date="2015-01-01")
def tarif_klassen_5_und_6(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    parameter_max_lohnsteuer_klasse_5_6: PiecewisePolynomialParamValue,
) -> float:
    """Lohnsteuer for Lohnsteuerklassen 5 and 6."""

    lohnsteuer_klasse_5_6 = basis_für_klassen_5_6(
        einkommen_y, einkommensteuer__parameter_einkommensteuertarif
    )
    max_lohnsteuer = piecewise_polynomial(
        einkommen_y, parameter_max_lohnsteuer_klasse_5_6
    )
    min_lohnsteuer = (
        einkommensteuer__parameter_einkommensteuertarif.rates[0, 1] * einkommen_y
    )
    return np.minimum(np.maximum(min_lohnsteuer, lohnsteuer_klasse_5_6), max_lohnsteuer)


@policy_function(start_date="2015-01-01")
def betrag_m(
    steuerklasse: int,
    basistarif: float,
    splittingtarif: float,
    tarif_klassen_5_und_6: float,
) -> float:
    """Withholding tax on earnings (Lohnsteuer)"""

    if steuerklasse == 1 or steuerklasse == 2 or steuerklasse == 4:
        out = basistarif
    elif steuerklasse == 3:
        out = splittingtarif
    else:
        out = tarif_klassen_5_und_6
    out = out / 12
    return max(out, 0.0)


@policy_function(start_date="2015-01-01")
def basistarif_mit_kinderfreibetrag(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    kinderfreibetrag_soli_y: float,
) -> float:
    """Lohnsteuer in the Basistarif deducting the Kindefreibetrag."""
    einkommen_abzüglich_kinderfreibetrag_soli = np.maximum(
        einkommen_y - kinderfreibetrag_soli_y, 0
    )
    return piecewise_polynomial(
        einkommen_abzüglich_kinderfreibetrag_soli,
        einkommensteuer__parameter_einkommensteuertarif,
    )


@policy_function(start_date="2015-01-01")
def splittingtarif_mit_kinderfreibetrag(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    kinderfreibetrag_soli_y: float,
) -> float:
    """Lohnsteuer in the Splittingtarif deducting the Kindefreibetrag."""
    einkommen_abzüglich_kinderfreibetrag_soli = np.maximum(
        einkommen_y - kinderfreibetrag_soli_y, 0
    )
    return 2 * piecewise_polynomial(
        einkommen_abzüglich_kinderfreibetrag_soli / 2,
        einkommensteuer__parameter_einkommensteuertarif,
    )


@policy_function(start_date="2015-01-01")
def tarif_klassen_5_und_6_mit_kinderfreibetrag(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    parameter_max_lohnsteuer_klasse_5_6: PiecewisePolynomialParamValue,
    kinderfreibetrag_soli_y: float,
) -> float:
    """Lohnsteuer for Lohnsteuerklassen 5 and 6 deducting the Kindefreibetrag."""
    einkommen_abzüglich_kinderfreibetrag_soli = np.maximum(
        einkommen_y - kinderfreibetrag_soli_y, 0
    )

    lohnsteuer_klasse_5_6 = basis_für_klassen_5_6(
        einkommen_abzüglich_kinderfreibetrag_soli,
        einkommensteuer__parameter_einkommensteuertarif,
    )
    max_lohnsteuer = piecewise_polynomial(
        einkommen_abzüglich_kinderfreibetrag_soli, parameter_max_lohnsteuer_klasse_5_6
    )
    min_lohnsteuer = (
        einkommensteuer__parameter_einkommensteuertarif.rates[0, 1]
        * einkommen_abzüglich_kinderfreibetrag_soli
    )
    return np.minimum(np.maximum(min_lohnsteuer, lohnsteuer_klasse_5_6), max_lohnsteuer)


@policy_function(start_date="2015-01-01")
def betrag_mit_kinderfreibetrag_m(
    steuerklasse: int,
    basistarif_mit_kinderfreibetrag: float,
    splittingtarif_mit_kinderfreibetrag: float,
    tarif_klassen_5_und_6_mit_kinderfreibetrag: float,
) -> float:
    """Withholding tax taking child allowances into account.

    Same as betrag_m, but with an alternative income definition that
    takes child allowance into account. Important only for calculation
    of Solidaritätszuschlag on Lohnsteuer!
    """
    if steuerklasse == 1 or steuerklasse == 2 or steuerklasse == 4:
        out = basistarif_mit_kinderfreibetrag
    elif steuerklasse == 3:
        out = splittingtarif_mit_kinderfreibetrag
    else:
        out = tarif_klassen_5_und_6_mit_kinderfreibetrag
    out = out / 12
    return max(out, 0.0)


@policy_function(start_date="2015-01-01")
def betrag_soli_y(
    betrag_mit_kinderfreibetrag_y: float,
    solidaritätszuschlag__parameter_solidaritätszuschlag: PiecewisePolynomialParamValue,
) -> float:
    """Solidarity surcharge on Lohnsteuer (withholding tax on earnings)."""

    return piecewise_polynomial(
        x=betrag_mit_kinderfreibetrag_y,
        parameters=solidaritätszuschlag__parameter_solidaritätszuschlag,
    )


@policy_function(start_date="2015-01-01")
def kinderfreibetrag_soli_y(
    steuerklasse: int,
    einkommensteuer__kinderfreibetrag_y: int,
) -> float:
    """Child Allowance (Kinderfreibetrag) for Lohnsteuer-Soli.

    For the purpose of Soli on Lohnsteuer, Steuerklasse 1/2/3 gets twice the child
    benefit, Steuerklasse 4 gets the child benefit once, and Steuerklasse 5/6 gets
    nothing.
    """

    if steuerklasse == 1 or steuerklasse == 2 or steuerklasse == 3:
        out = 2 * einkommensteuer__kinderfreibetrag_y
    elif steuerklasse == 4:
        out = einkommensteuer__kinderfreibetrag_y
    else:
        out = 0
    return out
