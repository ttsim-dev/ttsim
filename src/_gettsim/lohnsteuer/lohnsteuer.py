"""Withholding tax on earnings (Lohnsteuer)."""

from __future__ import annotations

from _gettsim.einkommensteuer.einkommensteuer import einkommensteuertarif
from _gettsim.solidaritätszuschlag.solidaritätszuschlag import (
    solidaritätszuschlagstarif,
)
from ttsim import PiecewisePolynomialParamValue, policy_function


@policy_function(vectorization_strategy="loop")
def betrag_m(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    steuerklasse: int,
    lohnst_params: dict,
) -> float:
    """
    Withholding tax on earnings (Lohnsteuer).

    """
    return lohnsteuerformel(
        einkommen_y,
        einkommensteuer__parameter_einkommensteuertarif,
        lohnst_params,
        steuerklasse,
    )


@policy_function(vectorization_strategy="loop")
def betrag_mit_kinderfreibetrag_m(
    einkommen_y: float,
    kinderfreibetrag_soli_y: float,
    steuerklasse: int,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    lohnst_params: dict,
) -> float:
    """Withholding tax taking child allowances into account.

    Same as betrag_m, but with an alternative income definition that
    takes child allowance into account. Important only for calculation
    of soli on Lohnsteuer!
    """

    eink = max(einkommen_y - kinderfreibetrag_soli_y, 0)

    return lohnsteuerformel(
        eink,
        einkommensteuer__parameter_einkommensteuertarif,
        lohnst_params,
        steuerklasse,
    )


@policy_function(vectorization_strategy="loop")
def betrag_soli_y(
    betrag_mit_kinderfreibetrag_y: float,
    solidaritätszuschlag__parameter_solidaritätszuschlag: PiecewisePolynomialParamValue,
) -> float:
    """Solidarity surcharge on Lohnsteuer (withholding tax on earnings)."""

    return solidaritätszuschlagstarif(
        steuer_pro_person=betrag_mit_kinderfreibetrag_y,
        parameter_solidaritätszuschlag=solidaritätszuschlag__parameter_solidaritätszuschlag,
    )


@policy_function(vectorization_strategy="loop")
def kinderfreibetrag_soli_y(
    steuerklasse: int,
    einkommensteuer__kinderfreibetrag_y: int,
) -> float:
    """Child Allowance (Kinderfreibetrag) for Lohnsteuer-Soli.

    For the purpose of Soli on Lohnsteuer, Steuerklasse 1/2/3 gets twice the child
    benefit, Steuerklasse 4 gets the child benefit once, and Steuerklasse 5/6 gets
    nothing.
    """

    if steuerklasse in {1, 2, 3}:
        out = 2 * einkommensteuer__kinderfreibetrag_y
    elif steuerklasse == 4:
        out = einkommensteuer__kinderfreibetrag_y
    else:
        out = 0
    return out


def lohnsteuerformel(
    einkommen_y: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    lohnst_params: dict,
    steuerklasse: int,
) -> float:
    """
    Calculates Lohnsteuer (withholding tax on earnings), paid monthly by the employer on
    behalf of the employee. Apply the income tax tariff, but individually and with
    different exemptions, determined by the 'steuerklasse'. Source: §39b EStG

    Calculation is differentiated by steuerklasse

    1,2,4: Standard tariff (§32a (1) EStG) 3: Splitting tariff (§32a (5) EStG) 5,6: Take
    twice the difference between applying the tariff on 5/4 and 3/4 of taxable income.
    Tax rate may not be lower than the starting statutory one.

    """

    lohnsteuer_basistarif = einkommensteuertarif(
        einkommen_y, einkommensteuer__parameter_einkommensteuertarif
    )
    lohnsteuer_splittingtarif = 2 * einkommensteuertarif(
        einkommen_y / 2, einkommensteuer__parameter_einkommensteuertarif
    )
    lohnsteuer_5_6_basis = basis_für_klassen_5_6(
        einkommen_y=einkommen_y,
        parameter_einkommensteuertarif=einkommensteuer__parameter_einkommensteuertarif,
    )

    lohnsteuer_grenze_1 = basis_für_klassen_5_6(
        einkommen_y=lohnst_params["einkommensgrenzen"][1],
        parameter_einkommensteuertarif=einkommensteuer__parameter_einkommensteuertarif,
    )
    max_lohnsteuer = (
        lohnsteuer_grenze_1
        + (einkommen_y - lohnst_params["einkommensgrenzen"][1])
        * einkommensteuer__parameter_einkommensteuertarif.rates[0, 3]
    )
    lohnsteuer_grenze_2 = basis_für_klassen_5_6(
        einkommen_y=lohnst_params["einkommensgrenzen"][2],
        parameter_einkommensteuertarif=einkommensteuer__parameter_einkommensteuertarif,
    )
    lohnsteuer_zw_grenze_2_3 = (
        lohnst_params["einkommensgrenzen"][3] - lohnst_params["einkommensgrenzen"][2]
    ) * einkommensteuer__parameter_einkommensteuertarif.rates[0, 3]
    lohnsteuer_klasse5_6_tmp = lohnsteuer_grenze_2 + lohnsteuer_zw_grenze_2_3

    if einkommen_y < lohnst_params["einkommensgrenzen"][1]:
        lohnsteuer_klasse5_6 = lohnsteuer_5_6_basis
    elif (
        lohnst_params["einkommensgrenzen"][1]
        <= einkommen_y
        < lohnst_params["einkommensgrenzen"][2]
    ):
        lohnsteuer_klasse5_6 = min(
            max_lohnsteuer,
            basis_für_klassen_5_6(
                einkommen_y=einkommen_y,
                parameter_einkommensteuertarif=einkommensteuer__parameter_einkommensteuertarif,
            ),
        )
    elif (
        lohnst_params["einkommensgrenzen"][2]
        <= einkommen_y
        < lohnst_params["einkommensgrenzen"][3]
    ):
        lohnsteuer_klasse5_6 = (
            lohnsteuer_grenze_2
            + (einkommen_y - lohnst_params["einkommensgrenzen"][2])
            * einkommensteuer__parameter_einkommensteuertarif.rates[0, 3]
        )
    else:
        lohnsteuer_klasse5_6 = (
            lohnsteuer_klasse5_6_tmp
            + (einkommen_y - lohnst_params["einkommensgrenzen"][3])
            * einkommensteuer__parameter_einkommensteuertarif.rates[0, 4]
        )

    if steuerklasse in {1, 2, 4}:
        out = lohnsteuer_basistarif
    elif steuerklasse == 3:
        out = lohnsteuer_splittingtarif
    else:
        out = lohnsteuer_klasse5_6

    out = out / 12

    return max(out, 0.0)


def basis_für_klassen_5_6(
    einkommen_y: float, parameter_einkommensteuertarif: PiecewisePolynomialParamValue
) -> float:
    """Calculate base for Lohnsteuer for steuerklasse 5 and 6, by applying
    obtaining twice the difference between applying the factors 1.25 and 0.75
    to the lohnsteuer payment. There is a also a minimum amount, which is checked
    afterwards.

    §39 b Absatz 2 Satz 7 (part 1):

        Jahreslohnsteuer die sich aus dem Zweifachen des Unterschiedsbetrags zwischen
        dem Steuerbetrag für das Eineinviertelfache und dem Steuerbetrag für das
        Dreiviertelfache des zu versteuernden Jahresbetrags nach § 32a Absatz 1 ergibt;
        die Jahreslohnsteuer beträgt jedoch mindestens 14 Prozent des zu versteuernden
        Jahresbetrags.

    """

    return max(
        2
        * (
            einkommensteuertarif(einkommen_y * 1.25, parameter_einkommensteuertarif)
            - einkommensteuertarif(einkommen_y * 0.75, parameter_einkommensteuertarif)
        ),
        einkommen_y * parameter_einkommensteuertarif.rates[0, 1],
    )
