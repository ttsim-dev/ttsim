"""Regularly employed."""

from ttsim import policy_function


@policy_function(end_date="2003-03-31", leaf_name="regulär_beschäftigt")
def regulär_beschäftigt_vor_midijob(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    minijob_grenze: float,
) -> bool:
    """Regular employment check until March 2003.

    Employees earning more than the minijob threshold, are subject to all ordinary
    income and social insurance contribution regulations. In gettsim we call these
    regular employed.

    """
    out = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        >= minijob_grenze
    )
    return out


@policy_function(start_date="2003-04-01", leaf_name="regulär_beschäftigt")
def regulär_beschäftigt_mit_midijob(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    geringfügige_einkommen_params: dict,
) -> bool:
    """Regular employment check since April 2003.

    Employees earning more than the midijob threshold, are subject to all ordinary
    income and social insurance contribution regulations. In gettsim we call these
    regular employed.

    """
    out = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        >= geringfügige_einkommen_params["grenzen_m"]["midijob"]
    )
    return out
