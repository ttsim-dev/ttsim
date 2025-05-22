"""Tax allowances for the elderly."""

from __future__ import annotations

from ttsim import policy_function
from ttsim.config import numpy_or_jax as np


@policy_function(end_date="2004-12-31", leaf_name="altersfreibetrag_y")
def altersfreibetrag_y_bis_2004(
    alter: int,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y: float,
    altersentlastungsbetrag_altersgrenze: int,
    maximaler_altersentlastungsbetrag_einheitlich: float,
    altersentlastungsquote_einheitlich: float,
) -> float:
    """Calculate tax deduction allowance for elderly until 2004."""
    altersgrenze = altersentlastungsbetrag_altersgrenze
    weiteres_einkommen = max(
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y,
        0.0,
    )
    if alter > altersgrenze:
        out = min(
            altersentlastungsquote_einheitlich
            * (
                einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
                + weiteres_einkommen
            ),
            maximaler_altersentlastungsbetrag_einheitlich,
        )
    else:
        out = 0.0

    return out


@policy_function(
    start_date="2005-01-01",
    leaf_name="altersfreibetrag_y",
    vectorization_strategy="loop",
)
def altersfreibetrag_y_ab_2005(
    alter: int,
    geburtsjahr: int,
    sozialversicherung__geringfügig_beschäftigt: bool,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y: float,
    altersentlastungsbetrag_altersgrenze: int,
    maximaler_altersentlastungsbetrag_gestaffelt: dict[int, float],
    altersentlastungsquote_gestaffelt: dict[int, float],
) -> float:
    """Calculate tax deduction allowance for elderly since 2005."""
    bins = sorted(maximaler_altersentlastungsbetrag_gestaffelt.keys())
    if geburtsjahr <= 1939:
        selected_bin = 1940
    else:
        selected_bin = bins[
            np.searchsorted(np.asarray([*bins, np.inf]), geburtsjahr, side="right") - 1
        ]

    out_max = maximaler_altersentlastungsbetrag_gestaffelt[selected_bin]

    einkommen_lohn = (
        0
        if sozialversicherung__geringfügig_beschäftigt
        else einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
    )
    weiteres_einkommen = max(
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y,
        0.0,
    )
    out_quote = altersentlastungsquote_gestaffelt[selected_bin] * (
        einkommen_lohn + weiteres_einkommen
    )

    if alter > altersentlastungsbetrag_altersgrenze:
        out = min(out_quote, out_max)
    else:
        out = 0.0

    return out
