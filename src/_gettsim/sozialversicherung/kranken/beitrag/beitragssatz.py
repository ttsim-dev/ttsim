"""Contribution rate for health insurance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import params_function, policy_function

if TYPE_CHECKING:
    from ttsim.typing import RawParamsRequiringConversion


@params_function(end_date="2005-06-30", leaf_name="parameter_beitragssatz")
def parameter_beitragssatz_uniform(
    raw_parameter_beitragssatz: RawParamsRequiringConversion,
) -> float:
    """Parameter for health insurance contribution rate before reunification."""
    return raw_parameter_beitragssatz["value"]


@params_function(
    start_date="2005-07-01", end_date="2008-12-31", leaf_name="parameter_beitragssatz"
)
def parameter_beitragssatz_mit_zusatz_und_sonderbeitrag(
    raw_parameter_beitragssatz: RawParamsRequiringConversion,
) -> dict:
    """Parameter for health insurance contribution rate after reunification."""
    return raw_parameter_beitragssatz


@policy_function(
    end_date="2005-06-30",
)
def beitragssatz_arbeitnehmer(
    parameter_beitragssatz: float,
) -> float:
    """Employee's health insurance contribution rate until June 2005.

    Basic split between employees and employers.
    """

    return parameter_beitragssatz / 2


@policy_function(
    end_date="2005-06-30",
)
def beitragssatz_arbeitnehmer_jahresanfang(
    parameter_beitragssatz_jahresanfang: float,
) -> float:
    """Employee's health insurance contribution rate for the beginning of the year until
    June 2005.

    Basic split between employees and employers.
    """
    return parameter_beitragssatz_jahresanfang / 2


@policy_function(
    start_date="2005-07-01",
    end_date="2008-12-31",
    leaf_name="beitragssatz_arbeitnehmer",
)
def beitragssatz_arbeitnehmer_mittlerer_kassenspezifischer_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz: dict,
) -> float:
    """Employee's health insurance contribution rate.

    From July 2005 until December 2008. The contribution rates consists of a general
    rate (split equally between employers and employees, differs across sickness funds)
    and a top-up rate, which is fully paid by employees.
    """
    return parameter_beitragssatz["mean_allgemein"] / 2 + zusatzbeitragssatz


@policy_function(
    start_date="2005-07-01",
    end_date="2008-12-31",
    leaf_name="beitragssatz_arbeitnehmer_jahresanfang",
)
def beitragssatz_arbeitnehmer_jahresanfang_mittlerer_kassenspezifischer_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz_jahresanfang: dict,
) -> float:
    """Employee's health insurance contribution rate at the beginning of the year.

    From July 2005 until December 2008. The contribution rates consists of a general
    rate (split equally between employers and employees, differs across sickness funds)
    and a top-up rate, which is fully paid by employees.
    """
    return (
        parameter_beitragssatz_jahresanfang["mean_allgemein"] / 2 + zusatzbeitragssatz
    )


@policy_function(
    start_date="2009-01-01",
    end_date="2018-12-31",
    leaf_name="beitragssatz_arbeitnehmer",
)
def beitragssatz_arbeitnehmer_einheitlicher_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz: dict,
) -> float:
    """Employee's health insurance contribution rate.

    From January 2009 until December 2018. The contribution rates consists of a general
    rate (split equally between employers and employees, same for all sickness funds)
    and a top-up rate, which is fully paid by employees.
    """
    return parameter_beitragssatz["allgemein"] / 2 + zusatzbeitragssatz


@policy_function(
    start_date="2009-01-01",
    end_date="2018-12-31",
    leaf_name="beitragssatz_arbeitnehmer_jahresanfang",
)
def beitragssatz_arbeitnehmer_jahresanfang_einheitlicher_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz_jahresanfang: dict,
) -> float:
    """Employee's health insurance contribution rate at the beginning of the year.

    From January 2009 until December 2018. The contribution rates consists of a general
    rate (split equally between employers and employees, same for all sickness funds)
    and a top-up rate, which is fully paid by employees.
    """
    return parameter_beitragssatz_jahresanfang["allgemein"] / 2 + zusatzbeitragssatz


@policy_function(
    start_date="2019-01-01",
    leaf_name="beitragssatz_arbeitnehmer",
)
def beitragssatz_arbeitnehmer_paritätischer_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz: dict,
) -> float:
    """Employee's health insurance contribution rate.

    Since 2019. Zusatzbeitrag is split equally between employers and employees.

    Parameters
    ----------
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    return (parameter_beitragssatz["allgemein"] + zusatzbeitragssatz) / 2


@policy_function(
    start_date="2019-01-01",
    leaf_name="beitragssatz_arbeitnehmer_jahresanfang",
)
def beitragssatz_arbeitnehmer_jahresanfang_paritätischer_zusatzbeitrag(
    zusatzbeitragssatz: float,
    parameter_beitragssatz_jahresanfang: dict,
) -> float:
    """Employee's health insurance contribution rate at the beginning of the year.

    Zusatzbeitrag is now split equally between employers and employees.
    """
    return (parameter_beitragssatz_jahresanfang["allgemein"] + zusatzbeitragssatz) / 2


@policy_function(
    end_date="2008-12-31",
    leaf_name="beitragssatz_arbeitgeber",
)
def beitragssatz_arbeitgeber_mittlerer_kassenspezifischer(
    parameter_beitragssatz: float,
) -> float:
    """Employer's health insurance contribution rate.

    Until 2008, the top-up contribution rate (Zusatzbeitrag) was not considered.
    """

    return parameter_beitragssatz / 2


@policy_function(
    end_date="2008-12-31",
    leaf_name="beitragssatz_arbeitgeber_jahresanfang",
)
def beitragssatz_arbeitgeber_jahresanfang_mittlerer_kassenspezifischer(
    parameter_beitragssatz_jahresanfang: float,
) -> float:
    """Employer's health insurance contribution rate at the begging of the year.

    Until 2008, the top-up contribution rate (Zusatzbeitrag) was not considered.
    """

    return parameter_beitragssatz_jahresanfang / 2


@policy_function(
    start_date="2009-01-01",
    end_date="2018-12-31",
    leaf_name="beitragssatz_arbeitgeber",
)
def beitragssatz_arbeitgeber_einheitlicher_zusatzbeitrag(
    parameter_beitragssatz: dict,
) -> float:
    """Employer's health insurance contribution rate.

    From 2009 until 2018, the contribution rate was uniform for all health insurers,
    Zusatzbeitrag irrelevant.
    """

    return parameter_beitragssatz["allgemein"] / 2


@policy_function(
    start_date="2009-01-01",
    end_date="2018-12-31",
    leaf_name="beitragssatz_arbeitgeber_jahresanfang",
)
def beitragssatz_arbeitgeber_jahresanfang_einheitlicher_zusatzbeitrag(
    parameter_beitragssatz_jahresanfang: dict,
) -> float:
    """Employer's health insurance contribution rate at the beginning of the year.

    From 2009 until 2018, the contribution rate was uniform for all health insurers,
    Zusatzbeitrag irrelevant.
    """

    return parameter_beitragssatz_jahresanfang["allgemein"] / 2


@policy_function(
    start_date="2019-01-01",
    leaf_name="beitragssatz_arbeitgeber",
)
def beitragssatz_arbeitgeber_paritätischer_zusatzbeitrag(
    beitragssatz_arbeitnehmer: float,
) -> float:
    """Employer's health insurance contribution rate.

    Since 2019, the full contribution rate is now split equally between employers and
    employees.
    """
    return beitragssatz_arbeitnehmer


@policy_function(
    start_date="2019-01-01",
    leaf_name="beitragssatz_arbeitgeber_jahresanfang",
)
def beitragssatz_arbeitgeber_jahresanfang_paritätischer_zusatzbeitrag(
    beitragssatz_arbeitnehmer_jahresanfang: float,
) -> float:
    """Employer's health insurance contribution rate at the beginning of the year.

    Since 2019, the full contribution rate is now split equally between employers and
    employees.
    """
    return beitragssatz_arbeitnehmer_jahresanfang


@policy_function(
    start_date="2005-07-01",
    end_date="2014-12-31",
    leaf_name="zusatzbeitragssatz",
)
def zusatzbeitragssatz_von_sonderbeitrag(
    parameter_beitragssatz: dict,
) -> float:
    """Health insurance top-up (Zusatzbeitrag) rate until December 2014."""

    return parameter_beitragssatz["sonderbeitrag"]


@policy_function(
    start_date="2015-01-01",
    leaf_name="zusatzbeitragssatz",
)
def zusatzbeitragssatz_von_mean_zusatzbeitrag(
    parameter_beitragssatz: dict,
) -> float:
    """Health insurance top-up rate (Zusatzbeitrag) since January 2015."""

    return parameter_beitragssatz["mean_zusatzbeitrag"]
