"""Regular pathway."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import params_function, policy_function
from ttsim.config import numpy_or_jax as np
from ttsim.piecewise_polynomial import get_piecewise_parameters, piecewise_polynomial

if TYPE_CHECKING:
    from ttsim import PiecewisePolynomialTTSIMParam, RawTTSIMParam


@policy_function(
    end_date="2007-04-19",
    leaf_name="altersgrenze",
    vectorization_strategy="not_required",
)
def altersgrenze_ohne_staffelung(regelaltersgrenze: float) -> float:
    """Normal retirement age (NRA).

    NRA is the same for every birth cohort.

    The Regelaltersrente cannot be claimed earlier than at the NRA, i.e. the NRA does
    not serve as reference for calculating deductions. However, it serves as reference
    for calculating gains in the Zugangsfakor in case of later retirement.

    Does not check for eligibility for this pathway into retirement.
    """
    return regelaltersgrenze


@policy_function(
    start_date="2007-04-20", leaf_name="altersgrenze", vectorization_strategy="loop"
)
def altersgrenze_mit_staffelung(
    geburtsjahr: int, piecewise_params_regelaltersgrenze: PiecewisePolynomialTTSIMParam
) -> float:
    """Normal retirement age (NRA).

    NRA differs by birth cohort.

    The Regelaltersrente cannot be claimed earlier than at the NRA, i.e. the NRA does
    not serve as reference for calculating deductions. However, it serves as reference
    for calculating gains in the Zugangsfakor in case of later retirement.

    Does not check for eligibility for this pathway into retirement.
    """
    return piecewise_polynomial(
        x=geburtsjahr,
        parameters=piecewise_params_regelaltersgrenze,
    )


@policy_function()
def grundsätzlich_anspruchsberechtigt(
    sozialversicherung__rente__mindestwartezeit_erfüllt: bool,
) -> bool:
    """Determining the eligibility for the Regelaltersrente."""

    return sozialversicherung__rente__mindestwartezeit_erfüllt


@params_function(start_date="2007-04-20")
def piecewise_params_regelaltersgrenze(
    regelaltersgrenze_gestaffelt: RawTTSIMParam,
) -> PiecewisePolynomialTTSIMParam:
    """Parameters for the piecewise polynomial Regelaltersgrenze (staggered by birth
    cohort)."""

    full_spec = regelaltersgrenze_gestaffelt.copy()

    max_birthyear_old_regime = full_spec.pop("max_birthyear_old_regime")
    min_birthyear_new_regime = full_spec.pop("min_birthyear_new_regime")
    entry_age_old_regime = full_spec.pop("entry_age_old_regime")
    entry_age_new_regime = full_spec.pop("entry_age_new_regime")

    parameter_dict: dict[int, dict[str, float]] = {}

    # Old regime
    parameter_dict[0] = {
        "lower_threshold": -np.inf,
        "upper_threshold": max_birthyear_old_regime,
        "rate_linear": 0,
        "intercept_at_lower_threshold": entry_age_old_regime,
    }

    last_retirement_age = entry_age_old_regime
    last_birthyear = max_birthyear_old_regime

    for idx, (birthyear, spec) in enumerate(sorted(full_spec.items()), start=1):
        current_retirement_age = spec["base_age"] + spec["months_to_add"] / 12

        # Piecewise spec for one birth year between old and new regime
        parameter_dict[idx] = {
            "lower_threshold": last_birthyear,
            "upper_threshold": birthyear,
            "rate_linear": current_retirement_age - last_retirement_age,
        }

        last_retirement_age = current_retirement_age
        last_birthyear = birthyear

    # Transition to new regime
    transition_idx = len(full_spec) + 1
    parameter_dict[transition_idx] = {
        "lower_threshold": last_birthyear,
        "upper_threshold": min_birthyear_new_regime,
        "rate_linear": entry_age_new_regime - last_retirement_age,
    }

    # New regime
    parameter_dict[transition_idx + 1] = {
        "lower_threshold": min_birthyear_new_regime,
        "upper_threshold": np.inf,
        "rate_linear": 0,
    }

    return get_piecewise_parameters(
        leaf_name="piecewise_params_regelaltersgrenze",
        func_type="piecewise_linear",
        parameter_dict=parameter_dict,
    )
