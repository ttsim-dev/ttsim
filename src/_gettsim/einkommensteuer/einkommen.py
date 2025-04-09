"""Einkommen.

Einkommen are Einkünfte minus Sonderausgaben, Vorsorgeaufwendungen, außergewöhnliche
Belastungen and sonstige Abzüge."""

from ttsim import policy_function


@policy_function()
def gesamteinkommen_y(
    einkommensteuer__einkünfte__gesamtbetrag_der_einkünfte_y_sn: float,
    einkommensteuer__abzüge__betrag_y_sn: float,
) -> float:
    """Gesamteinkommen without Kinderfreibetrag on tax unit level."""
    out = (
        einkommensteuer__einkünfte__gesamtbetrag_der_einkünfte_y_sn
        - einkommensteuer__abzüge__betrag_y_sn
    )

    return max(out, 0.0)
