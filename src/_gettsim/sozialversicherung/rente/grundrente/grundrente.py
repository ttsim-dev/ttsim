from __future__ import annotations

from ttsim import RoundingSpec, piecewise_polynomial, policy_function


@policy_function(
    rounding_spec=RoundingSpec(
        base=0.01, direction="nearest", reference="§ 123 SGB VI Abs. 1"
    ),
    start_date="2021-01-01",
)
def betrag_m(basisbetrag_m: float, anzurechnendes_einkommen_m: float) -> float:
    """Calculate Grundrentenzuschlag (additional monthly pensions payments resulting
    from Grundrente)

    Parameters
    ----------
    basisbetrag_m
        See :func:`basisbetrag_m`.
    anzurechnendes_einkommen_m
        See :func:`anzurechnendes_einkommen_m`.

    Returns
    -------

    """
    out = basisbetrag_m - anzurechnendes_einkommen_m
    return max(out, 0.0)


@policy_function(start_date="2021-01-01")
def einkommen_m(
    einkommensteuer__einkünfte__sonstige__renteneinkünfte_vorjahr_m: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m: float,
    einkommensteuer__einkünfte__aus_kapitalvermögen__betrag_m: float,
) -> float:
    """Calculate total income relevant for Grundrentenzuschlag before deductions are
    subtracted.

    Some notes:

    - The Grundrentenzuschlag (in previous years) is not part of the relevant income and
      does not lower the Grundrentenzuschlag (reference: § 97a Abs. 2 S. 7 SGB VI).
    - The Deutsche Rentenversicherung uses the income of the year two to three years ago
      to be able to use administrative data on this income for the calculation: "It can
      be assumed that the tax office regularly has the data two years after the end of
      the assessment period, which can be retrieved from the pension insurance."
    - Warning: Currently, earnings of dependent work and pensions are based on the last
      year, and other income on the current year instead of the year two years ago to
      avoid the need for several new input variables.
    - Warning: Freibeträge for income are currently not considered as `freibeträge_y`
      depends on pension income through
      `sozialversicherung__kranken__beitrag__betrag_versicherter_m` ->
      `vorsorgeaufw` -> `freibeträge`

    Reference: § 97a Abs. 2 S. 1 SGB VI

    Parameters
    ----------
    einkommensteuer__einkünfte__sonstige__renteneinkünfte_vorjahr_m
        See :func:`einkommensteuer__einkünfte__sonstige__renteneinkünfte_vorjahr_m`.
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m
        See :func:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m`.
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m
        See :func:`einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m`.
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m
        See :func:`einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m`.
    einkommensteuer__einkünfte__aus_kapitalvermögen__betrag_m
        See :func:`einkommensteuer__einkünfte__aus_kapitalvermögen__betrag_m`.

    Returns
    -------

    """

    # Sum income over different income sources.
    return (
        einkommensteuer__einkünfte__sonstige__renteneinkünfte_vorjahr_m
        + einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m  # income from self-employment
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m  # rental income
        + einkommensteuer__einkünfte__aus_kapitalvermögen__betrag_m
    )


@policy_function(
    rounding_spec=RoundingSpec(
        base=0.01, direction="nearest", reference="§ 123 SGB VI Abs. 1"
    ),
    start_date="2021-01-01",
    vectorization_strategy="loop",
)
def anzurechnendes_einkommen_m(
    einkommen_m_ehe: float,
    familie__anzahl_personen_ehe: int,
    sozialversicherung__rente__altersrente__rentenwert: float,
    ges_rente_params: dict,
) -> float:
    """Calculate income which is deducted from Grundrentenzuschlag.

    Apply allowances. There are upper and lower thresholds for singles and
    couples. 60% of income between the upper and lower threshold is credited against
    the Grundrentenzuschlag. All the income above the upper threshold is credited
    against the Grundrentenzuschlag.

    Reference: § 97a Abs. 4 S. 2, 4 SGB VI

    Parameters
    ----------
    einkommen_m_ehe
        See :func:`einkommen_m_ehe`.
    familie__anzahl_personen_ehe
        See :func:`familie__anzahl_personen_ehe`.
    sozialversicherung__rente__altersrente__rentenwert
        See :func:`sozialversicherung__rente__altersrente__rentenwert`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.
    Returns
    -------

    """

    # Calculate relevant income following the crediting rules using the values for
    # singles and those for married subjects
    # Note: Thresholds are defined relativ to rentenwert which is implemented by
    # dividing the income by rentenwert and multiply rentenwert to the result.
    if familie__anzahl_personen_ehe == 2:
        einkommensanr_params = ges_rente_params[
            "grundrente_anzurechnendes_einkommen_mit_partner"
        ]
    else:
        einkommensanr_params = ges_rente_params[
            "grundrente_anzurechnendes_einkommen_ohne_partner"
        ]

    out = (
        piecewise_polynomial(
            x=einkommen_m_ehe / sozialversicherung__rente__altersrente__rentenwert,
            parameters=einkommensanr_params,
        )
        * sozialversicherung__rente__altersrente__rentenwert
    )

    return out


@policy_function(
    rounding_spec=RoundingSpec(
        base=0.01, direction="nearest", reference="§ 123 SGB VI Abs. 1"
    ),
    start_date="2021-01-01",
)
def basisbetrag_m(
    mean_entgeltpunkte_zuschlag: float,
    bewertungszeiten_monate: int,
    sozialversicherung__rente__altersrente__rentenwert: float,
    sozialversicherung__rente__altersrente__zugangsfaktor: float,
    ges_rente_params: dict,
) -> float:
    """Calculate additional monthly pensions payments resulting from Grundrente, without
    taking into account income crediting rules.

    The Zugangsfaktor is limited to 1 and considered Grundrentezeiten
    are limited to 35 years (420 months).

    Parameters
    ----------
    mean_entgeltpunkte_zuschlag
        See :func:`mean_entgeltpunkte_zuschlag`.
    bewertungszeiten_monate
        See basic input variable
        :ref:`bewertungszeiten_monate <bewertungszeiten_monate>`.
    sozialversicherung__rente__altersrente__rentenwert
        See :func:`sozialversicherung__rente__altersrente__rentenwert`.
    sozialversicherung__rente__altersrente__zugangsfaktor
        See :func:`sozialversicherung__rente__altersrente__zugangsfaktor`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------

    """

    # Winsorize Bewertungszeiten and Zugangsfaktor at maximum values
    bewertungszeiten_monate_wins = min(
        bewertungszeiten_monate,
        ges_rente_params["grundrente_berücksichtigte_wartezeit"]["max"],
    )
    ges_rente_zugangsfaktor_wins = min(
        sozialversicherung__rente__altersrente__zugangsfaktor,
        ges_rente_params["grundrente_maximaler_zugangsfaktor"],
    )

    return (
        mean_entgeltpunkte_zuschlag
        * bewertungszeiten_monate_wins
        * sozialversicherung__rente__altersrente__rentenwert
        * ges_rente_zugangsfaktor_wins
    )


@policy_function(start_date="2021-01-01")
def durchschnittliche_entgeltpunkte(
    mean_entgeltpunkte: float, bewertungszeiten_monate: int
) -> float:
    """Compute average number of Entgeltpunkte earned per month of
    Grundrentenbewertungszeiten.

    Parameters
    ----------
    mean_entgeltpunkte
        See basic input variable
        :ref:`mean_entgeltpunkte <mean_entgeltpunkte>`.
    bewertungszeiten_monate
        See basic input variable
        :ref:`bewertungszeiten_monate <bewertungszeiten_monate>`.

    Returns
    -------

    """
    if bewertungszeiten_monate > 0:
        out = mean_entgeltpunkte / bewertungszeiten_monate

    # Return 0 if bewertungszeiten_monate is 0. Then, mean_entgeltpunkte should be 0, too.
    else:
        out = 0

    return out


@policy_function(
    rounding_spec=RoundingSpec(
        base=0.0001,
        direction="nearest",
        reference="§76g SGB VI Abs. 4 Nr. 4",
    ),
    start_date="2021-01-01",
)
def höchstbetrag_m(
    grundrentenzeiten_monate: int,
    ges_rente_params: dict,
) -> float:
    """Calculate the maximum allowed number of average Entgeltpunkte (per month) after
    adding bonus of Entgeltpunkte for a given number of Grundrentenzeiten.

    Parameters
    ----------
    grundrentenzeiten_monate
        See basic input variable :ref:`grundrentenzeiten_monate <grundrentenzeiten_monate>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------

    """
    # Calculate number of months above minimum threshold
    months_above_thresh = (
        min(
            grundrentenzeiten_monate,
            ges_rente_params["grundrente_berücksichtigte_wartezeit"]["max"],
        )
        - ges_rente_params["grundrente_berücksichtigte_wartezeit"]["min"]
    )

    # Calculate höchstwert
    return (
        ges_rente_params["grundrente_höchstwert_der_entgeltpunkte"]["base"]
        + ges_rente_params["grundrente_höchstwert_der_entgeltpunkte"]["increment"]
        * months_above_thresh
    )


@policy_function(
    rounding_spec=RoundingSpec(
        base=0.0001,
        direction="nearest",
        reference="§ 123 SGB VI Abs. 1",
    ),
    start_date="2021-01-01",
    vectorization_strategy="loop",
)
def mean_entgeltpunkte_zuschlag(
    durchschnittliche_entgeltpunkte: float,
    höchstbetrag_m: float,
    grundrentenzeiten_monate: int,
    ges_rente_params: dict,
) -> float:
    """Calculate additional Entgeltpunkte for pensioner.

    In general, the average of monthly Entgeltpunkte earnd in Grundrentenzeiten is
    doubled, or extended to the individual Höchstwert if doubling would exceed the
    Höchstwert. Then, the value is multiplied by 0.875.

    Legal reference: § 76g SGB VI

    Parameters
    ----------
    durchschnittliche_entgeltpunkte
        See :func:`durchschnittliche_entgeltpunkte`.
    höchstbetrag_m
        See :func:`höchstbetrag_m`.
    grundrentenzeiten_monate
        See basic input variable :ref:`grundrentenzeiten_monate <grundrentenzeiten_monate>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------

    """

    # Return 0 if Grundrentenzeiten below minimum
    if (
        grundrentenzeiten_monate
        < ges_rente_params["grundrente_berücksichtigte_wartezeit"]["min"]
    ):
        out = 0.0
    else:
        # Case 1: Entgeltpunkte less than half of Höchstwert
        if durchschnittliche_entgeltpunkte <= (0.5 * höchstbetrag_m):
            out = durchschnittliche_entgeltpunkte

        # Case 2: Entgeltpunkte more than half of Höchstwert, but below Höchstwert
        elif durchschnittliche_entgeltpunkte < höchstbetrag_m:
            out = höchstbetrag_m - durchschnittliche_entgeltpunkte

        # Case 3: Entgeltpunkte above Höchstwert
        elif durchschnittliche_entgeltpunkte > höchstbetrag_m:
            out = 0.0

    # Multiply additional Engeltpunkte by factor
    return out * ges_rente_params["grundrente_bonusfaktor"]


@policy_function(start_date="2021-01-01")
def grundsätzlich_anspruchsberechtigt(
    grundrentenzeiten_monate: int,
    ges_rente_params: dict,
) -> bool:
    """Whether person has accumulated enough insured years to be eligible.

    Parameters
    ----------
    grundrentenzeiten_monate
        See :func:`grundrentenzeiten_monate`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------

    """
    return (
        grundrentenzeiten_monate
        >= ges_rente_params["grundrente_berücksichtigte_wartezeit"]["min"]
    )
