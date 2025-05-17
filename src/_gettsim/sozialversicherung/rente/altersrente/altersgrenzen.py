"""Age thresholds for public pension eligibility."""

from ttsim import policy_function


@policy_function(
    end_date="2011-12-31",
    leaf_name="altersgrenze_abschlagsfrei",
    vectorization_strategy="loop",
)
def altersgrenze_abschlagsfrei_mit_arbeitsl_frauen_ohne_besond_langj(
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze: float,
    sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__für_frauen__altersgrenze: float,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Full retirement age after eligibility checks, assuming eligibility for
    Regelaltersrente.

    Age at which pension can be claimed without deductions. This age is smaller or equal
    to the normal retirement age (FRA<=NRA) and depends on personal characteristics as
    gender, insurance duration, health/disability, employment status.
    """

    out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze
    if sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt:
        out = min(
            [out, sozialversicherung__rente__altersrente__für_frauen__altersgrenze]
        )
    if sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt:
        out = min(
            [
                out,
                sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze,
            ]
        )
    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = min(
            [out, sozialversicherung__rente__altersrente__langjährig__altersgrenze]
        )

    return out


@policy_function(
    start_date="2012-01-01",
    end_date="2017-12-31",
    leaf_name="altersgrenze_abschlagsfrei",
    vectorization_strategy="loop",
)
def altersgrenze_abschlagsfrei_mit_frauen_arbeitsl_besond_langj(
    sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__für_frauen__altersgrenze: float,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze: float,
    sozialversicherung__rente__altersrente__besonders_langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__besonders_langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Full retirement age after eligibility checks, assuming eligibility for
    Regelaltersrente.

    Age at which pension can be claimed without deductions. This age is smaller or equal
    to the normal retirement age (FRA<=NRA) and depends on personal characteristics as
    gender, insurance duration, health/disability, employment status.

    Starting in 2012, the pension for the very long term insured (Altersrente für
    besonders langjährig Versicherte) is introduced. Policy becomes inactive in 2018
    because then all potential beneficiaries of the Rente wg. Arbeitslosigkeit and
    Rente für Frauen have reached the normal retirement age.
    """

    out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze
    if sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt:
        out = min(
            [out, sozialversicherung__rente__altersrente__für_frauen__altersgrenze]
        )
    if sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt:
        out = min(
            [
                out,
                sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze,
            ]
        )
    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = min(
            [out, sozialversicherung__rente__altersrente__langjährig__altersgrenze]
        )
    if sozialversicherung__rente__altersrente__besonders_langjährig__grundsätzlich_anspruchsberechtigt:
        out = min(
            [
                out,
                sozialversicherung__rente__altersrente__besonders_langjährig__altersgrenze,
            ]
        )

    return out


@policy_function(
    start_date="2018-01-01",
    leaf_name="altersgrenze_abschlagsfrei",
    vectorization_strategy="loop",
)
def altersgrenze_abschlagsfrei_mit_besond_langj_ohne_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__besonders_langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__besonders_langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Full retirement age after eligibility checks, assuming eligibility for
    Regelaltersrente.

    Age at which pension can be claimed without deductions. This age is smaller or equal
    to the normal retirement age (FRA<=NRA) and depends on personal characteristics as
    gender, insurance duration, health/disability, employment status.
    """

    out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze
    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = min(
            [out, sozialversicherung__rente__altersrente__langjährig__altersgrenze]
        )
    if sozialversicherung__rente__altersrente__besonders_langjährig__grundsätzlich_anspruchsberechtigt:
        out = min(
            [
                out,
                sozialversicherung__rente__altersrente__besonders_langjährig__altersgrenze,
            ]
        )

    return out


@policy_function(
    end_date="2017-12-31",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="loop",
)
def altersgrenze_vorzeitig_mit_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze_vorzeitig: float,
    sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__für_frauen__altersgrenze_vorzeitig: float,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze_vorzeitig: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Earliest possible retirement age after checking for eligibility.

    Early retirement age depends on personal characteristics as gender, insurance
    duration, health/disability, employment status. Policy becomes inactive in 2018
    because then all potential beneficiaries of the Rente wg. Arbeitslosigkeit and Rente
    für Frauen have reached the normal retirement age.
    """
    frauen_vorzeitig = (
        sozialversicherung__rente__altersrente__für_frauen__altersgrenze_vorzeitig
    )

    arbeitsl_vorzeitig = sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze_vorzeitig

    langjährig_vorzeitig = (
        sozialversicherung__rente__altersrente__langjährig__altersgrenze_vorzeitig
    )

    out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze

    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = langjährig_vorzeitig
    if sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt:
        out = min([out, frauen_vorzeitig])
    if sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt:
        out = min([out, arbeitsl_vorzeitig])

    return out


@policy_function(start_date="2018-01-01", leaf_name="altersgrenze_vorzeitig")
def altersgrenze_vorzeitig_ohne_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze_vorzeitig: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Earliest possible retirement age after checking for eligibility.

    Early retirement age depends on personal characteristics as gender, insurance
    duration, health/disability, employment status.
    """

    out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze

    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = sozialversicherung__rente__altersrente__langjährig__altersgrenze_vorzeitig
    else:
        out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze

    return out


@policy_function(
    end_date="2017-12-31", leaf_name="vorzeitig_grundsätzlich_anspruchsberechtigt"
)
def vorzeitig_grundsätzlich_anspruchsberechtigt_mit_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt: bool,
) -> bool:
    """Eligibility for some form ofearly retirement.

    Can only be claimed if eligible for "Rente für langjährig Versicherte" or "Rente für
    Frauen" or "Rente für Arbeitslose" (or -not yet implemented - for disabled). Policy
    becomes inactive in 2018 because then all potential beneficiaries of the Rente wg.
    Arbeitslosigkeit and Rente für Frauen have reached the normal retirement age.
    """

    return (
        sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt
        or sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt
        or sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt
    )


@policy_function(
    start_date="2018-01-01", leaf_name="vorzeitig_grundsätzlich_anspruchsberechtigt"
)
def vorzeitig_grundsätzlich_anspruchsberechtigt_vorzeitig_ohne_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
) -> bool:
    """Eligibility for early retirement.

    Can only be claimed if eligible for "Rente für langjährig Versicherte".
    """

    return sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt


@policy_function(end_date="2017-12-31", leaf_name="referenzalter_abschlag")
def referenzalter_abschlag_mit_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze: float,
    sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__für_frauen__altersgrenze: float,
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Reference age for deduction calculation in case of early retirement
    (Zugangsfaktor).

    Normal retirement age if not eligible for early retirement. Policy becomes inactive
    in 2018 because then all potential beneficiaries of the Rente wg. Arbeitslosigkeit
    and Rente für Frauen have reached the normal retirement age.
    """
    if (
        sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt
        and sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt
        and sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt
    ):
        out = min(
            [
                sozialversicherung__rente__altersrente__für_frauen__altersgrenze,
                sozialversicherung__rente__altersrente__langjährig__altersgrenze,
                sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze,
            ]
        )
    elif (
        sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt
        and sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt
    ):
        out = min(
            [
                sozialversicherung__rente__altersrente__für_frauen__altersgrenze,
                sozialversicherung__rente__altersrente__langjährig__altersgrenze,
            ]
        )
    elif (
        sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt
        and sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt
    ):
        out = min(
            [
                sozialversicherung__rente__altersrente__langjährig__altersgrenze,
                sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze,
            ]
        )
    elif sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = sozialversicherung__rente__altersrente__langjährig__altersgrenze
    elif sozialversicherung__rente__altersrente__für_frauen__grundsätzlich_anspruchsberechtigt:
        out = sozialversicherung__rente__altersrente__für_frauen__altersgrenze
    elif sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__grundsätzlich_anspruchsberechtigt:
        out = (
            sozialversicherung__rente__altersrente__wegen_arbeitslosigkeit__altersgrenze
        )
    else:
        out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze

    return out


@policy_function(start_date="2018-01-01", leaf_name="referenzalter_abschlag")
def referenzalter_abschlag_ohne_rente_arbeitsl_frauen(
    sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__langjährig__altersgrenze: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
) -> float:
    """Reference age for deduction calculation in case of early retirement
    (Zugangsfaktor).

    Normal retirement age if not eligible for early retirement.
    """
    if sozialversicherung__rente__altersrente__langjährig__grundsätzlich_anspruchsberechtigt:
        out = sozialversicherung__rente__altersrente__langjährig__altersgrenze
    else:
        out = sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze

    return out
