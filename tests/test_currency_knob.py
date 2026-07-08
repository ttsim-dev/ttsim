"""The currency knob and build-time parameter conversion (GEP 10)."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pytest
from mettsim import middle_earth

from ttsim import MainTarget, OrigPolicyObjects, main
from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.policy_environment import (
    _get_one_param,
    function_like_converter_output_in_run_currency,
)
from ttsim.interface_dag_elements.specialized_environment import (
    _convert_function_like_converter_outputs,
)
from ttsim.tt import Unit, param_function
from ttsim.tt import units as units_module
from ttsim.tt.param_objects import PiecewisePolynomialParamValue, RawParam
from ttsim.tt.units import (
    UNSET_UNIT,
    base_currency,
    currency_conversion_factor,
    currency_family_root,
    register_currency,
    registered_base_currencies,
)

POLICY_DATE = datetime.date(2020, 1, 1)

_HEADER = {
    "name": {"de": "foo", "en": "foo"},
    "description": {"de": "foo", "en": "foo"},
}


def _policy_environment(backend, currency=None):
    return main(
        main_target=MainTarget.policy_environment,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=POLICY_DATE,
        backend=backend,
        currency=currency,
    )


def test_currency_defaults_to_registered_base(backend):
    assert (
        main(
            main_target=MainTarget.currency,
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            policy_date=POLICY_DATE,
            backend=backend,
        )
        == "CASTAR"
    )


def test_currency_override_threads_through_to_param_conversion(backend):
    """Overriding the knob changes how parameters are converted at build time."""
    base = _policy_environment(backend)["payroll_tax"][
        "wealth_threshold_for_reduced_tax_rate"
    ].value
    silver = _policy_environment(backend, currency="SILVER_PENNY")["payroll_tax"][
        "wealth_threshold_for_reduced_tax_rate"
    ].value
    assert silver == pytest.approx(base * 4)


def test_param_unchanged_in_base_currency(backend):
    """From the 2020 reform on, the threshold is legislated in castar."""
    env = _policy_environment(backend)
    threshold = env["payroll_tax"]["wealth_threshold_for_reduced_tax_rate"]
    assert threshold.value == pytest.approx(12500)


def test_param_converted_to_run_currency(backend):
    """A silver-penny run converts the castar-denominated threshold at build."""
    env = _policy_environment(backend, currency="SILVER_PENNY")
    threshold = env["payroll_tax"]["wealth_threshold_for_reduced_tax_rate"]
    # 12_500 castar = 50_000 silver pennies (silver_penny = castar / 4).
    assert threshold.value == pytest.approx(50000)


def test_changeover_is_a_pure_redenomination(backend):
    """The 2020 reform re-expresses the threshold; its real value is continuous."""
    for currency in ("CASTAR", "SILVER_PENNY"):
        before = main(
            main_target=MainTarget.policy_environment,
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            policy_date=datetime.date(2019, 12, 31),
            backend=backend,
            currency=currency,
        )["payroll_tax"]["wealth_threshold_for_reduced_tax_rate"]
        after = _policy_environment(backend, currency=currency)["payroll_tax"][
            "wealth_threshold_for_reduced_tax_rate"
        ]
        assert before.value == pytest.approx(after.value)


# ----------------------------------------------------------------------------
# Token-driven conversion of the parameter shapes
# ----------------------------------------------------------------------------


def _load(
    leaf_name: str,
    spec: Any,
    policy_date: datetime.date,
    currency: str | None,
) -> Any:
    param = _get_one_param(
        leaf_name=leaf_name,
        spec=spec,
        policy_date=policy_date,
        xnp=np,
        currency=currency,
    )
    assert param is not None
    return param


def _scalar_spec(**header):
    return {
        **_HEADER,
        "unit": "SILVER_PENNY",
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0},
        **header,
    }


def test_scalar_conversion_reads_currency_off_the_declaration():
    param = _load(
        leaf_name="threshold",
        spec=_scalar_spec(),
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    # 100 silver pennies = 25 castar (silver_penny = castar / 4).
    assert param.value == pytest.approx(25.0)


def test_scalar_conversion_is_a_no_op_in_the_source_currency():
    param = _load(
        leaf_name="threshold",
        spec=_scalar_spec(),
        policy_date=POLICY_DATE,
        currency="SILVER_PENNY",
    )
    assert param.value == pytest.approx(100.0)


def test_scalar_conversion_is_a_no_op_without_a_run_currency():
    param = _load(
        leaf_name="threshold",
        spec=_scalar_spec(),
        policy_date=POLICY_DATE,
        currency=None,
    )
    assert param.value == pytest.approx(100.0)


def test_entry_level_override_writes_a_changeover():
    """Old entries in silver pennies, entries from the reform date in castar."""
    spec = {
        **_HEADER,
        "unit": "SILVER_PENNY",
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0},
        datetime.date(2020, 1, 1): {"value": 25.0, "unit": "CASTAR"},
    }
    before = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2019, 12, 31),
        currency="CASTAR",
    )
    after = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2020, 1, 1),
        currency="CASTAR",
    )
    # A pure redenomination: identical magnitudes in the run currency.
    assert before.value == pytest.approx(25.0)
    assert after.value == pytest.approx(25.0)


def test_unit_forward_fills_across_a_gap():
    """A dated entry without ``unit:`` inherits the most recent earlier unit.

    The reproducer from the GEP-10 design discussion: the 1990 entry omits
    ``unit:`` and there is no top-level fallback, yet it resolves — to the
    silver penny declared at 1900, forward-filled.
    """
    spec = {
        **_HEADER,
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0, "unit": "SILVER_PENNY"},
        datetime.date(1990, 1, 1): {"value": 130.0},
        datetime.date(2000, 1, 1): {"value": 25.0, "unit": "CASTAR"},
    }
    # Active at 1995 is the 1990 entry; its unit forward-fills to silver penny.
    same = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(1995, 6, 1),
        currency="SILVER_PENNY",
    )
    assert same.value == pytest.approx(130.0)
    # 130 silver pennies = 32.5 castar (silver_penny = castar / 4).
    converted = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(1995, 6, 1),
        currency="CASTAR",
    )
    assert converted.value == pytest.approx(32.5)


def test_unit_forward_fill_carries_a_changeover_onward():
    """A date-specific unit becomes the new seed: later unit-less entries inherit
    it, not the top-level/original declaration."""
    spec = {
        **_HEADER,
        "unit": "SILVER_PENNY",
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0},
        datetime.date(2000, 1, 1): {"value": 25.0, "unit": "CASTAR"},
        datetime.date(2010, 1, 1): {"value": 30.0},
    }
    # Active at 2015 is the 2010 entry; it inherits castar (from 2000), not the
    # top-level silver penny — so a castar run is a no-op.
    no_op = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2015, 1, 1),
        currency="CASTAR",
    )
    assert no_op.value == pytest.approx(30.0)
    # 30 castar = 120 silver pennies; had it wrongly reverted to the seed, this
    # would read 30.
    converted = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2015, 1, 1),
        currency="SILVER_PENNY",
    )
    assert converted.value == pytest.approx(120.0)


def test_unit_resolution_never_backfills_from_a_later_entry():
    """Resolution only ever looks backward. A gap with no earlier declaration and
    no top-level stays unset (the mandatory-unit gate fires downstream); it does
    not borrow a unit from a future entry."""
    spec = {
        **_HEADER,
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0},
        datetime.date(2000, 1, 1): {"value": 25.0, "unit": "CASTAR"},
    }
    param = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(1950, 1, 1),
        currency="CASTAR",
    )
    assert param.unit is UNSET_UNIT


def test_updates_previous_may_cross_a_unit_change_at_the_authors_risk():
    """The changeover guard is gone (GEP 10): ``updates_previous`` (a value merge)
    and a unit restatement are independent mechanisms. Combining them merges
    old-currency leaves forward under the new unit — silently wrong by the
    conversion factor and invisible to dimensional checks (same dimension, only
    the scale differs). A documented sharp edge, the author's responsibility: a
    unit-change entry should restate its values in full."""
    spec = {
        **_HEADER,
        "unit": "SILVER_PENNY",
        "type": "dict",
        datetime.date(1900, 1, 1): {"a": 100.0, "b": 8.0},
        datetime.date(2020, 1, 1): {
            "a": 25.0,
            "unit": "CASTAR",
            "updates_previous": True,
        },
    }
    param = _load(
        leaf_name="amounts",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    # `a` was restated in castar; `b` is silently carried from the silver-penny
    # era yet now labelled castar — the sharp edge, no longer an error.
    assert param.value == {"a": pytest.approx(25.0), "b": pytest.approx(8.0)}


def test_mapping_unit_restatement_must_be_complete():
    """A dated restatement of a per-leaf ``unit:`` mapping must cover every leaf:
    ttsim replaces the mapping wholesale rather than merging, so a partial
    restatement would silently leave some leaves on the old currency."""
    spec = {
        **_HEADER,
        "type": "dict",
        datetime.date(1900, 1, 1): {
            "child_amount_y": 100.0,
            "max_age": 18,
            "unit": {"child_amount_y": "SILVER_PENNY_PER_YEAR", "max_age": "YEARS"},
        },
        datetime.date(2000, 1, 1): {
            "child_amount_y": 25.0,
            "max_age": 18,
            "unit": {"child_amount_y": "CASTAR_PER_YEAR"},
        },
    }
    with pytest.raises(UnitDefinitionError, match="every leaf"):
        _load(
            leaf_name="schedule",
            spec=spec,
            policy_date=POLICY_DATE,
            currency="CASTAR",
        )


def test_piecewise_conversion_currency_input_axis():
    """An income schedule: bounds and intercepts scale, slopes are invariant."""
    spec = {
        **_HEADER,
        "input_unit": "SILVER_PENNY_PER_YEAR",
        "output_unit": "SILVER_PENNY_PER_YEAR",
        "type": "piecewise_linear",
        datetime.date(1900, 1, 1): {
            "intervals": [
                {"interval": "(-inf, 0)", "slope": 0.0, "intercept": 0},
                {"interval": "[0, 100)", "slope": 0.1},
                {"interval": "[100, inf)", "slope": 0.3},
            ]
        },
    }
    param = _load(
        leaf_name="tax_schedule",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    # 100 silver pennies = 25 castar.
    assert param.value.thresholds[2] == pytest.approx(25.0)
    # Slopes (penny per penny) are dimensionless and invariant.
    assert param.value.coefficients[1, 0] == pytest.approx(0.1)
    assert param.value.coefficients[2, 0] == pytest.approx(0.3)
    # The intercept at the 100-penny kink: 0.1 * 100 pennies = 10 pennies
    # = 2.5 castar.
    assert param.value.intercepts[2] == pytest.approx(2.5)


def test_piecewise_conversion_non_currency_input_axis():
    """An area schedule: bounds stay, intercepts and slopes scale."""
    spec = {
        **_HEADER,
        "input_unit": "HECTARE",
        "output_unit": "SILVER_PENNY_PER_YEAR",
        "type": "piecewise_linear",
        datetime.date(1900, 1, 1): {
            "intervals": [
                {"interval": "(-inf, 10)", "slope": 0.0, "intercept": 0},
                {"interval": "[10, inf)", "slope": 100.0},
            ]
        },
    }
    param = _load(
        leaf_name="tax_schedule",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    # Bounds are in hectares: untouched.
    assert param.value.thresholds[1] == pytest.approx(10.0)
    # Slopes are pennies per hectare: scaled to castar per hectare.
    assert param.value.coefficients[1, 0] == pytest.approx(25.0)


def test_lookup_table_conversion_scales_values_only():
    spec = {
        **_HEADER,
        "input_unit": "YEARS",
        "output_unit": "SILVER_PENNY_PER_MONTH",
        "type": "sparse_to_consecutive_int_lookup_table",
        datetime.date(1900, 1, 1): {
            1950: 2000.0,
            "min_int_in_table": 1900,
            "max_int_in_table": 2050,
        },
    }
    param = _load(
        leaf_name="max_amount_m_fam_by_policy_year",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    assert param.value.look_up(2000) == pytest.approx(500.0)


def test_lookup_table_rejects_currency_input_axis():
    spec = {
        **_HEADER,
        "input_unit": "SILVER_PENNY",
        "output_unit": "SILVER_PENNY_PER_MONTH",
        "type": "sparse_to_consecutive_int_lookup_table",
        datetime.date(1900, 1, 1): {
            0: 1.0,
            "min_int_in_table": 0,
            "max_int_in_table": 10,
        },
    }
    with pytest.raises(UnitDefinitionError, match="integer-keyed"):
        _load(
            leaf_name="table",
            spec=spec,
            policy_date=POLICY_DATE,
            currency="CASTAR",
        )


def test_dict_param_converts_currency_leaves_only():
    spec = {
        **_HEADER,
        "unit": {"child_amount_y": "SILVER_PENNY_PER_YEAR", "max_age": "YEARS"},
        "type": "dict",
        datetime.date(1900, 1, 1): {"child_amount_y": 100.0, "max_age": 18},
    }
    param = _load(
        leaf_name="schedule",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    assert param.value["child_amount_y"] == pytest.approx(25.0)
    assert param.value["max_age"] == 18


def test_require_converter_converts_currency_leaves_only():
    """Each leaf converts per its own token, at any nesting depth (GEP 10)."""
    spec = {
        **_HEADER,
        "unit": {
            "amounts": {
                "base_m": "SILVER_PENNY_PER_MONTH",
                "supplement_m": "SILVER_PENNY_PER_MONTH",
            },
            "bounds": {"min_age": "YEARS", "max_age": "YEARS"},
        },
        "type": "require_converter",
        datetime.date(1900, 1, 1): {
            "amounts": {"base_m": 100.0, "supplement_m": 40.0},
            "bounds": {"min_age": 0, "max_age": 18},
        },
    }
    param = _load(
        leaf_name="raw_child_rate",
        spec=spec,
        policy_date=POLICY_DATE,
        currency="CASTAR",
    )
    assert param.value["amounts"]["base_m"] == pytest.approx(25.0)
    assert param.value["amounts"]["supplement_m"] == pytest.approx(10.0)
    assert param.value["bounds"]["max_age"] == 18


def test_function_like_converter_output_converts_polynomial_per_order():
    """A require_converter's piecewise output scales per polynomial order (GEP 10 S3).

    The order-``j`` coefficient must scale by ``f_out / f_in**j`` — the slope
    invariant, the quadratic by ``1 / f_in`` — not by one uniform factor, which
    would silently mis-state the schedule (the audit's polynomial case).
    """
    schedule = PiecewisePolynomialParamValue(
        thresholds=np.array([0.0, 100.0]),
        intercepts=np.array([0.0, 5.0]),
        coefficients=np.array([[1.0, 0.5], [0.3, 0.0]]),
    )
    converted = function_like_converter_output_in_run_currency(
        value=schedule,
        input_unit="SILVER_PENNY_PER_YEAR",
        output_unit="SILVER_PENNY_PER_YEAR",
        run_currency="CASTAR",
        xnp=np,
        leaf_name="tax_schedule",
    )
    # f_in = f_out = silver_penny in castar = 0.25.
    assert converted.thresholds[1] == pytest.approx(25.0)  # input axis x 0.25
    assert converted.intercepts[1] == pytest.approx(1.25)  # output axis x 0.25
    assert converted.coefficients[0, 0] == pytest.approx(1.0)  # slope: invariant
    # Quadratic: f_out / f_in**2 = 0.25 / 0.0625 = 4 (NOT the uniform 0.25).
    assert converted.coefficients[0, 1] == pytest.approx(2.0)


def test_require_converter_with_axes_is_left_raw_for_its_converter():
    """A function-like require_converter is not uniform-scaled at load (GEP 10).

    Its raw blob passes through unchanged; the conversion happens later, on the
    converter's typed output, per axis.
    """
    spec = {
        **_HEADER,
        "input_unit": "SILVER_PENNY",
        "output_unit": "SILVER_PENNY_PER_YEAR",
        "type": "require_converter",
        datetime.date(1900, 1, 1): {"top_rate": 0.2, "ceiling": 100},
    }
    param = _load(
        leaf_name="raw_schedule", spec=spec, policy_date=POLICY_DATE, currency="CASTAR"
    )
    # The ceiling is a currency amount but is NOT scaled here.
    assert param.value["ceiling"] == 100
    assert param.input_unit is not UNSET_UNIT


def test_homogeneous_require_converter_producing_a_schedule_is_rejected():
    """Uniform-scaling a structured schedule is rejected; declare axes (GEP 10 S3)."""

    @param_function(unit=Unit.DIMENSIONLESS)
    def schedule(raw_schedule: Any) -> Any:
        return raw_schedule

    raw = RawParam(value={"a": 1.0}, unit="SILVER_PENNY")
    outputs = {
        "schedule": PiecewisePolynomialParamValue(
            thresholds=np.array([0.0]),
            intercepts=np.array([0.0]),
            coefficients=np.array([[0.0]]),
        )
    }
    with pytest.raises(UnitDefinitionError, match="input_unit"):
        _convert_function_like_converter_outputs(
            outputs=outputs,
            params={"raw_schedule": raw},
            param_functions={"schedule": schedule},
            run_currency="CASTAR",
            xnp=np,
        )


def test_mapping_require_converter_producing_a_schedule_is_rejected():
    """A currency leaf in the mapping feeding a function-like output is
    rejected exactly as a homogeneous currency token (GEP 10)."""

    @param_function(unit=Unit.DIMENSIONLESS)
    def schedule(raw_schedule: Any) -> Any:
        return raw_schedule

    raw = RawParam(
        value={"ceiling": 1000.0, "top_rate": 0.2},
        unit={"ceiling": "SILVER_PENNY", "top_rate": "DIMENSIONLESS"},
    )
    outputs = {
        "schedule": PiecewisePolynomialParamValue(
            thresholds=np.array([0.0]),
            intercepts=np.array([0.0]),
            coefficients=np.array([[0.0]]),
        )
    }
    with pytest.raises(UnitDefinitionError, match="input_unit"):
        _convert_function_like_converter_outputs(
            outputs=outputs,
            params={"raw_schedule": raw},
            param_functions={"schedule": schedule},
            run_currency="CASTAR",
            xnp=np,
        )


def test_require_converter_unit_and_axes_are_mutually_exclusive():
    """A require_converter declares `unit:` xor axes, not both (GEP 10)."""
    with pytest.raises(UnitDefinitionError, match="not both"):
        RawParam(
            value={"a": 1.0},
            unit="SILVER_PENNY",
            input_unit="SILVER_PENNY",
        )


def test_unknown_annotation_is_rejected_at_load():
    # An annotation the vocabulary does not know fails loudly at load time
    # (issue #121) — nothing is ever silently scaled or left unconverted.
    spec = _scalar_spec(unit="Euros")
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        _load(
            leaf_name="threshold",
            spec=spec,
            policy_date=POLICY_DATE,
            currency="CASTAR",
        )


# --------------------------------------------------------------------------
# Currency families: two packages' registrations coexist in one process
# (GEP 10). mettsim's CASTAR family is registered by the import above; these
# tests register a second family and restore the bookkeeping afterwards (the
# pint definitions cannot be removed, but with the bookkeeping restored they
# are inert, and re-registration tolerates consistent leftovers).
# --------------------------------------------------------------------------


@pytest.fixture
def second_currency_family():
    saved_currencies = set(units_module._registered_currencies)
    saved_roots = dict(units_module._currency_family_root)
    saved_tokens = set(units_module._ALLOWED_UNIT_TOKENS)
    register_currency("GOLD_DRAGON", base=True)
    register_currency("COPPER_STAR", definition="GOLD_DRAGON / 56")
    yield
    units_module._registered_currencies.clear()
    units_module._registered_currencies.update(saved_currencies)
    units_module._currency_family_root.clear()
    units_module._currency_family_root.update(saved_roots)
    units_module._ALLOWED_UNIT_TOKENS.clear()
    units_module._ALLOWED_UNIT_TOKENS.update(saved_tokens)


def test_second_currency_family_coexists(second_currency_family):
    assert currency_family_root("COPPER_STAR") == "GOLD_DRAGON"
    assert currency_family_root("SILVER_PENNY") == "CASTAR"
    assert set(registered_base_currencies()) >= {"CASTAR", "GOLD_DRAGON"}


def test_base_currency_is_ambiguous_across_families(second_currency_family):
    with pytest.raises(UnitDefinitionError, match="no process-wide default"):
        base_currency()


def test_conversion_across_families_is_rejected(second_currency_family):
    # Both bases sit at factor 1 against the abstract CURRENCY reference, so
    # pint would relate them 1:1 — a silent wrong number the family check
    # turns into a loud error.
    with pytest.raises(UnitDefinitionError, match="No exchange rate connects"):
        currency_conversion_factor(
            source_currency="SILVER_PENNY", run_currency="GOLD_DRAGON"
        )


def test_default_currency_follows_the_policy_objects(second_currency_family, backend):
    # The mixed-process scenario: with two families registered, the default
    # run currency is read off the parameters' declarations, not the process
    # registry — mettsim's parameters are denominated in the CASTAR family.
    assert (
        main(
            main_target=MainTarget.currency,
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            policy_date=POLICY_DATE,
            backend=backend,
        )
        == "CASTAR"
    )
