"""Einkünfte according to §§ 13-24 EStG."""

from ttsim import policy_function


@policy_function(end_date="2008-12-31", leaf_name="gesamtbetrag_der_einkünfte_y")
def gesamtbetrag_der_einkünfte_y_mit_kapiteleinkünften(
    aus_selbstständiger_arbeit__betrag_y: float,
    aus_nichtselbstständiger_arbeit__betrag_y: float,
    aus_vermietung_und_verpachtung__betrag_y: float,
    sonstige__betrag_y: float,
    aus_kapitalvermögen__betrag_y: float,
) -> float:
    """Gesamtbetrag der Einkünfte (GdE) mit Kapitaleinkünften.

    Parameters
    ----------
    aus_selbstständiger_arbeit__betrag_y
        See :func:`aus_selbstständiger_arbeit__betrag_y`.
    aus_nichtselbstständiger_arbeit__betrag_y
        See :func:`aus_nichtselbstständiger_arbeit__betrag_y`.
    aus_vermietung_und_verpachtung__betrag_y
        See :func:`aus_vermietung_und_verpachtung__betrag_y`.
    sonstige__betrag_y
        See :func:`sonstige__betrag_y`.
    aus_kapitalvermögen__betrag_y
        See :func:`aus_kapitalvermögen__betrag_y`.

    Returns
    -------

    """
    out = (
        aus_selbstständiger_arbeit__betrag_y
        + aus_nichtselbstständiger_arbeit__betrag_y
        + aus_vermietung_und_verpachtung__betrag_y
        + sonstige__betrag_y
        + aus_kapitalvermögen__betrag_y
    )
    return out


@policy_function(start_date="2009-01-01", leaf_name="gesamtbetrag_der_einkünfte_y")
def gesamtbetrag_der_einkünfte_y_ohne_kapitaleinkünfte(
    aus_selbstständiger_arbeit__betrag_y: float,
    aus_nichtselbstständiger_arbeit__betrag_y: float,
    aus_vermietung_und_verpachtung__betrag_y: float,
    sonstige__betrag_y: float,
) -> float:
    """Gesamtbetrag der Einkünfte (GdE) ohne Kapitaleinkünften.

    Since 2009 capital income is not subject to normal taxation.
    Parameters
    ----------
    aus_selbstständiger_arbeit__betrag_y
        See :func:`aus_selbstständiger_arbeit__betrag_y`.
    aus_nichtselbstständiger_arbeit__betrag_y
        See :func:`aus_nichtselbstständiger_arbeit__betrag_y`.
    aus_vermietung_und_verpachtung__betrag_y
        See :func:`aus_vermietung_und_verpachtung__betrag_y`.
    sonstige__betrag_y
        See :func:`sonstige__betrag_y`.

    Returns
    -------

    """
    out = (
        aus_selbstständiger_arbeit__betrag_y
        + aus_nichtselbstständiger_arbeit__betrag_y
        + aus_vermietung_und_verpachtung__betrag_y
        + sonstige__betrag_y
    )
    return out
