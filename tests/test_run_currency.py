"""The statutory computation currency and the input/output conversion (GEP 10).

Parameters are never converted: computation runs in the policy date's statutory
currency, every parameter must be declared in it (the statutory guard), and
only columns convert — user data on the way in, currency-denominated results on
the way out. mettsim registers CASTAR (base), SILVER_PENNY, and the statutory
mapping (silver pennies through 2019, castar from 2020) on the import below.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Any

import numpy as np
import pytest
from mettsim import middle_earth

from ttsim import MainTarget, OrigPolicyObjects, main
from ttsim.exceptions_and_warnings import (
    PotentialCurrencyMismatchWarning,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.policy_environment import (
    _get_one_param,
)
from ttsim.interface_dag_elements.results import tree_with_unit_annotations
from ttsim.interface_dag_elements.warn_if import (
    statutory_currency_and_base_currency_differ,
)
from ttsim.tt.currencies import (
    _statutory_currencies,
    base_currency,
    currency_conversion_factor,
    isolated_currency_registration,
    register_currency,
    register_statutory_currencies,
    statutory_currency_for_date,
)
from ttsim.tt.param_objects import RawParam
from ttsim.tt.units import UNIT_REGISTRY, UNSET_UNIT

POLICY_DATE = datetime.date(2020, 1, 1)

_HEADER = {
    "name": {"de": "foo", "en": "foo"},
    "description": {"de": "foo", "en": "foo"},
}


def _policy_environment(backend, policy_date=POLICY_DATE):
    return main(
        main_target=MainTarget.policy_environment,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=policy_date,
        backend=backend,
    )


def test_data_currency_defaults_to_registered_base(backend):
    assert (
        main(
            main_target=MainTarget.data_currency,
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            policy_date=POLICY_DATE,
            backend=backend,
        )
        == "CASTAR"
    )


@pytest.mark.parametrize(
    ("policy_date", "expected"),
    [
        (datetime.date(2019, 12, 31), "SILVER_PENNY"),
        (datetime.date(2020, 1, 1), "CASTAR"),
    ],
)
def test_computation_currency_is_the_statutory_currency_at_the_policy_date(
    policy_date, expected, backend
):
    assert (
        main(
            main_target=MainTarget.computation_currency,
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            policy_date=policy_date,
            backend=backend,
        )
        == expected
    )


def test_parameters_keep_their_statutory_values(backend):
    """No scaling at build: each era's threshold is exactly the statute's number."""
    castar_era = _policy_environment(backend)["payroll_tax"][
        "wealth_threshold_for_reduced_tax_rate"
    ]
    penny_era = _policy_environment(backend, policy_date=datetime.date(2019, 12, 31))[
        "payroll_tax"
    ]["wealth_threshold_for_reduced_tax_rate"]
    assert castar_era.value == pytest.approx(12500)
    assert penny_era.value == pytest.approx(50000)


def test_changeover_is_a_pure_redenomination(backend):
    """The 2020 reform re-expresses the threshold; its real value is continuous."""
    penny_era = _policy_environment(backend, policy_date=datetime.date(2019, 12, 31))[
        "payroll_tax"
    ]["wealth_threshold_for_reduced_tax_rate"]
    castar_era = _policy_environment(backend)["payroll_tax"][
        "wealth_threshold_for_reduced_tax_rate"
    ]
    # 50_000 silver pennies = 12_500 castar (silver_penny = castar / 4).
    assert penny_era.value == pytest.approx(castar_era.value * 4)


# ----------------------------------------------------------------------------
# The statutory guard: every parameter's concrete currency must be the
# statutory currency at the policy date.
# ----------------------------------------------------------------------------


def _load(
    leaf_name: str,
    spec: Any,
    policy_date: datetime.date,
    computation_currency: str,
) -> Any:
    param = _get_one_param(
        leaf_name=leaf_name,
        spec=spec,
        policy_date=policy_date,
        xnp=np,
        computation_currency=computation_currency,
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


def test_statutory_currency_declaration_passes_untouched():
    param = _load(
        leaf_name="threshold",
        spec=_scalar_spec(),
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value == pytest.approx(100.0)


def test_non_statutory_currency_declaration_is_rejected():
    """A silver-penny value surviving past the changeover fails the guard:
    parameters are never converted, so the author must add a dated entry
    restating the value in the statutory currency."""
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="threshold",
            spec=_scalar_spec(),
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


def test_entry_level_override_writes_a_changeover():
    """Old entries in silver pennies, entries from the reform date in castar —
    each era loads in its own statutory currency, values untouched."""
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
        computation_currency="SILVER_PENNY",
    )
    after = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2020, 1, 1),
        computation_currency="CASTAR",
    )
    assert before.value == pytest.approx(100.0)
    assert after.value == pytest.approx(25.0)


def test_unit_forward_fills_across_a_gap():
    """A dated entry without ``unit:`` inherits the most recent earlier unit.

    The reproducer from the GEP-10 design discussion: the 1990 entry omits
    ``unit:`` and there is no top-level fallback, yet it resolves — to the
    silver penny declared at 1900, forward-filled. The guard proves it: the
    entry passes in a silver-penny era and fails against castar.
    """
    spec = {
        **_HEADER,
        "type": "scalar",
        datetime.date(1900, 1, 1): {"value": 100.0, "unit": "SILVER_PENNY"},
        datetime.date(1990, 1, 1): {"value": 130.0},
        datetime.date(2000, 1, 1): {"value": 25.0, "unit": "CASTAR"},
    }
    param = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(1995, 6, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value == pytest.approx(130.0)
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="threshold",
            spec=spec,
            policy_date=datetime.date(1995, 6, 1),
            computation_currency="CASTAR",
        )


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
    # top-level silver penny — so it passes against castar and fails against
    # the silver penny.
    param = _load(
        leaf_name="threshold",
        spec=spec,
        policy_date=datetime.date(2015, 1, 1),
        computation_currency="CASTAR",
    )
    assert param.value == pytest.approx(30.0)
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="threshold",
            spec=spec,
            policy_date=datetime.date(2015, 1, 1),
            computation_currency="SILVER_PENNY",
        )


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
        computation_currency="CASTAR",
    )
    assert param.unit is UNSET_UNIT


def test_updates_previous_combined_with_a_unit_restatement_is_rejected():
    """A dated entry cannot both merge values (``updates_previous``) and restate
    the unit. The merge would carry un-restated leaves (``b`` here) forward from
    the silver-penny era under the new castar label — a silent mis-scaling by the
    conversion factor, invisible to the statutory-currency guard. A unit change
    must restate the value in full."""
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
    with pytest.raises(UnitDefinitionError, match="carry un-restated leaves"):
        _load(
            leaf_name="amounts",
            spec=spec,
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


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
            computation_currency="CASTAR",
        )


def test_guard_checks_piecewise_axes():
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
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    # The statute's numbers, exactly as written.
    assert param.value.thresholds[2] == pytest.approx(100.0)
    assert param.value.coefficients[1, 0] == pytest.approx(0.1)
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="tax_schedule",
            spec=spec,
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


def test_guard_checks_dict_param_leaves():
    spec = {
        **_HEADER,
        "unit": {"child_amount_y": "SILVER_PENNY_PER_YEAR", "max_age": "YEARS"},
        "type": "dict",
        datetime.date(1900, 1, 1): {"child_amount_y": 100.0, "max_age": 18},
    }
    param = _load(
        leaf_name="schedule",
        spec=spec,
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value["child_amount_y"] == pytest.approx(100.0)
    assert param.value["max_age"] == 18
    assert isinstance(param.value["max_age"], int)
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="schedule",
            spec=spec,
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


def test_guard_walks_int_keyed_per_leaf_mappings():
    # GEP-3 allows int dict keys (e.g. satz_nach_kindanzahl); the per-leaf unit
    # walk must accept an int in the path rather than tripping the beartype claw
    # on a `tuple[str, ...]` annotation (defect #4, GEP 10).
    spec = {
        **_HEADER,
        "unit": {1: "SILVER_PENNY_PER_MONTH", 2: "SILVER_PENNY_PER_MONTH"},
        "type": "dict",
        datetime.date(1900, 1, 1): {1: 100.0, 2: 200.0},
    }
    param = _load(
        leaf_name="satz_nach_kindanzahl",
        spec=spec,
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value[1] == pytest.approx(100.0)
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="satz_nach_kindanzahl",
            spec=spec,
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


def test_guard_walks_nested_require_converter_mappings():
    """Per-leaf tokens are checked at any nesting depth (GEP 10)."""
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
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value["amounts"]["base_m"] == pytest.approx(100.0)
    assert param.value["bounds"]["max_age"] == 18
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _load(
            leaf_name="raw_child_rate",
            spec=spec,
            policy_date=POLICY_DATE,
            computation_currency="CASTAR",
        )


def test_require_converter_with_axes_is_left_raw_for_its_converter():
    """An axes-declaring require_converter passes its raw value through; the
    declared axes describe the built schedule for checking, never conversion."""
    spec = {
        **_HEADER,
        "input_unit": "SILVER_PENNY",
        "output_unit": "SILVER_PENNY_PER_YEAR",
        "type": "require_converter",
        datetime.date(1900, 1, 1): {"top_rate": 0.2, "ceiling": 100},
    }
    param = _load(
        leaf_name="raw_schedule",
        spec=spec,
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    assert param.value["ceiling"] == 100
    assert param.input_unit is not UNSET_UNIT


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
    with pytest.raises(UnitDefinitionError, match="cannot be a currency"):
        _load(
            leaf_name="table",
            spec=spec,
            policy_date=datetime.date(1950, 1, 1),
            computation_currency="SILVER_PENNY",
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
    # (issue #121) — nothing is ever silently mis-declared.
    spec = _scalar_spec(unit="Euros")
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        _load(
            leaf_name="threshold",
            spec=spec,
            policy_date=datetime.date(1950, 1, 1),
            computation_currency="SILVER_PENNY",
        )


# --------------------------------------------------------------------------
# A single interconvertible currency system plus the statutory mapping
# (GEP 10). Registration rules are exercised inside
# `isolated_currency_registration`, so nothing leaks.
# --------------------------------------------------------------------------


def test_base_currency_is_the_registered_base():
    assert base_currency() == "CASTAR"


def test_registered_currencies_are_interconvertible():
    # silver_penny = castar / 4, so the factors are reciprocal.
    assert currency_conversion_factor(
        source_currency="SILVER_PENNY", target_currency="CASTAR"
    ) == pytest.approx(0.25)
    assert currency_conversion_factor(
        source_currency="CASTAR", target_currency="SILVER_PENNY"
    ) == pytest.approx(4.0)


def test_currency_conversion_rejects_the_abstract_currency_token():
    # The agnostic CURRENCY token is a pint unit but not a registered currency;
    # conversion (and hence `data_currency=`) requires a concrete registered one.
    with pytest.raises(UnitDefinitionError, match="not a registered currency"):
        currency_conversion_factor(source_currency="CURRENCY", target_currency="CASTAR")


def test_annotated_results_label_a_parameter_in_the_statutory_currency():
    """`tree_with_unit_annotations` labels a requested parameter in the
    computation (statutory) currency and a computed column in the data currency,
    even when the two differ (CASTAR data over a silver-penny computation). A
    parameter and a column both resolve to the agnostic CURRENCY, so the label
    must follow the result category, not the resolved unit: the parameter's value
    is never converted, the column's is (GEP 10)."""
    with isolated_currency_registration():
        register_currency(name="CASTAR", base=True)
        register_currency(name="SILVER_PENNY", definition="CASTAR / 4")
        agnostic = UNIT_REGISTRY.parse_units("CURRENCY / month")
        annotated = tree_with_unit_annotations(
            tree={"a_param_m": 25.0, "a_column_m": np.array([1.0, 2.0])},
            raw_results__from_input_data={},
            raw_results__params={"a_param_m": 25.0},
            unit_checks__resolved_units={"a_param_m": agnostic, "a_column_m": agnostic},
            data_currency="CASTAR",
            computation_currency="SILVER_PENNY",
        )
        labels = {qname: leaf.unit.base for qname, leaf in annotated.items()}
        assert labels == {"a_param_m": "SILVER_PENNY", "a_column_m": "CASTAR"}


def test_a_second_base_currency_is_rejected():
    # CASTAR is already the base; the process has a single base currency.
    with (
        isolated_currency_registration(),
        pytest.raises(UnitDefinitionError, match="already the base"),
    ):
        register_currency(name="GOLD_DRAGON", base=True)


def test_definition_referencing_no_registered_currency_is_rejected():
    # A currency must be defined relative to an already-registered one; a
    # definition against the abstract CURRENCY reference alone would start a
    # second, unconnected base.
    with (
        isolated_currency_registration(),
        pytest.raises(UnitDefinitionError, match="no registered currency"),
    ):
        register_currency(name="FLOATING", definition="CURRENCY / 2")


def test_statutory_currency_follows_the_dated_mapping():
    assert statutory_currency_for_date(datetime.date(2019, 12, 31)) == "SILVER_PENNY"
    assert statutory_currency_for_date(datetime.date(2020, 1, 1)) == "CASTAR"
    assert statutory_currency_for_date(datetime.date(2025, 6, 1)) == "CASTAR"


def test_statutory_currency_is_undefined_before_the_first_entry():
    with isolated_currency_registration():
        _statutory_currencies[:] = [(datetime.date(1900, 1, 1), "SILVER_PENNY")]
        with pytest.raises(UnitDefinitionError, match="Extend the mapping"):
            statutory_currency_for_date(datetime.date(1899, 12, 31))


def test_statutory_currency_requires_a_registered_mapping():
    with isolated_currency_registration():
        _statutory_currencies.clear()
        with pytest.raises(UnitDefinitionError, match="No statutory-currency mapping"):
            statutory_currency_for_date(datetime.date(2020, 1, 1))


def test_registering_an_empty_statutory_mapping_is_rejected():
    with (
        isolated_currency_registration(),
        pytest.raises(UnitDefinitionError, match="at least one entry"),
    ):
        register_statutory_currencies({})


def test_statutory_mapping_must_reference_registered_currencies():
    with (
        isolated_currency_registration(),
        pytest.raises(UnitDefinitionError, match="not registered"),
    ):
        register_statutory_currencies({"1900-01-01": "GOLD_DRAGON"})


def test_a_different_statutory_mapping_is_rejected():
    # mettsim's mapping is registered; a conflicting one must not replace it.
    with (
        isolated_currency_registration(),
        pytest.raises(UnitDefinitionError, match="already registered"),
    ):
        register_statutory_currencies({"1900-01-01": "CASTAR"})


def test_re_registering_the_identical_statutory_mapping_is_tolerated():
    # A re-imported module registers the same mapping again — a no-op.
    with isolated_currency_registration():
        register_statutory_currencies(
            {"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"}
        )


def test_warns_when_statutory_currency_differs_from_default_data_currency():
    """A run whose statutory currency is not the base while the data currency
    sits at its default (the base) may hold data denominated in the wrong
    currency — the user gets a nudge."""
    with pytest.warns(PotentialCurrencyMismatchWarning, match="denominated"):
        statutory_currency_and_base_currency_differ(
            computation_currency="SILVER_PENNY",
            data_currency="CASTAR",
            policy_date=datetime.date(2019, 1, 1),
        )


def test_no_warning_when_statutory_currency_is_the_base():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        statutory_currency_and_base_currency_differ(
            computation_currency="CASTAR",
            data_currency="CASTAR",
            policy_date=datetime.date(2025, 1, 1),
        )


def test_no_warning_when_the_data_currency_is_set_off_the_base():
    # The user chose a data currency explicitly; nothing to nudge about.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        statutory_currency_and_base_currency_differ(
            computation_currency="SILVER_PENNY",
            data_currency="SILVER_PENNY",
            policy_date=datetime.date(2019, 1, 1),
        )
