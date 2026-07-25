from __future__ import annotations

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim.middle_earth import UNIT_SYSTEM

from ttsim.interface_dag_elements.currency import (
    CurrencyConversion,
    input_data_in_computation_currency,
)
from ttsim.interface_dag_elements.specialized_environment import (
    _add_derived_functions,
)
from ttsim.tt import TTSIMUnit, policy_input


def _spec_env(policy_environment, input_data__flat, grouping_levels=()):
    """The specialized environment the currency conversion reads units off."""
    return _add_derived_functions(
        qname_env_without_tree_logic=policy_environment,
        tt_targets={},
        input_columns={dt.qname_from_tree_path(p) for p in input_data__flat},
        grouping_levels=grouping_levels,
    )


def test_pint_tagged_currency_input_is_stripped_and_rescaled():
    """A pint-tagged column is crossed into the data currency and loses its tag.

    A wealth column handed in as silver pennies rides along castar data; four
    silver pennies make one castar, so the magnitudes are quartered.
    """
    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wealth",): UNIT_SYSTEM.registry.Quantity(
            numpy.array([4.0, 8.0]), "SILVER_PENNY"
        ),
    }
    out = input_data_in_computation_currency(
        input_data__flat=input_data__flat,
        specialized_environment__without_tree_logic_and_with_derived_functions={},
        data_currency="CASTAR",
        computation_currency="CASTAR",
        unit_system=UNIT_SYSTEM,
    )
    assert not isinstance(out[("wealth",)], UNIT_SYSTEM.registry.Quantity)
    assert list(numpy.asarray(out[("wealth",)])) == pytest.approx([1.0, 2.0])


def test_untagged_input_converts_by_declared_unit():
    """Untagged data converts iff the column's declared unit carries a currency.

    Data arrive in the data currency; ``wealth`` is declared in currency and is
    crossed into the computation currency, while the non-currency ``age`` is
    handed back untouched.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> float:
        pass

    @policy_input(unit=TTSIMUnit.YEARS)
    def age() -> int:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wealth",): numpy.array([4.0, 8.0]),
        ("age",): numpy.array([30, 40]),
    }
    out = input_data_in_computation_currency(
        input_data__flat=input_data__flat,
        specialized_environment__without_tree_logic_and_with_derived_functions=(
            _spec_env(
                policy_environment={"wealth": wealth, "age": age},
                input_data__flat=input_data__flat,
                grouping_levels=("hh",),
            )
        ),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
        unit_system=UNIT_SYSTEM,
    )
    assert list(numpy.asarray(out[("wealth",)])) == pytest.approx([1.0, 2.0])
    assert list(numpy.asarray(out[("age",)])) == [30, 40]


def test_derived_input_ignores_siblings_a_derivation_cannot_produce():
    """`income_hh` aggregates the declared non-currency `income`; the currency
    flow `income_m` shares the base name but no derivation produces `income_hh`
    from it (a time suffix is only ever rebased, never dropped), so the column
    must not convert."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def income_m() -> float:
        pass

    @policy_input(unit=TTSIMUnit.YEARS)
    def income() -> int:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("income_hh",): numpy.array([4.0, 4.0]),
    }
    out = input_data_in_computation_currency(
        input_data__flat=input_data__flat,
        specialized_environment__without_tree_logic_and_with_derived_functions=(
            _spec_env(
                policy_environment={"income_m": income_m, "income": income},
                input_data__flat=input_data__flat,
                grouping_levels=("hh",),
            )
        ),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
        unit_system=UNIT_SYSTEM,
    )
    assert list(numpy.asarray(out[("income_hh",)])) == pytest.approx([4.0, 4.0])


def test_derived_input_converts_via_its_stub():
    """A derived name provided as data converts via its `PolicyInput` stub.

    ``wage_m_hh`` has no declaration of its own; the stub minted from the
    declared ``wage_m`` carries ``CURRENCY_PER_MONTH_PER_HH`` (aggregation
    never adds or removes the currency component), so the column converts.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def wage_m() -> float:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wage_m_hh",): numpy.array([4.0, 4.0]),
    }
    out = input_data_in_computation_currency(
        input_data__flat=input_data__flat,
        specialized_environment__without_tree_logic_and_with_derived_functions=(
            _spec_env(
                policy_environment={"wage_m": wage_m},
                input_data__flat=input_data__flat,
                grouping_levels=("hh",),
            )
        ),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
        unit_system=UNIT_SYSTEM,
    )
    assert list(numpy.asarray(out[("wage_m_hh",)])) == pytest.approx([1.0, 1.0])


def test_time_variant_input_converts_via_its_stub():
    """Data at a time-converted name converts via its `PolicyInput` stub.

    ``wage_y`` has no declaration of its own; the stub minted from the
    declared ``wage_m`` carries ``CURRENCY_PER_YEAR``, so the column converts.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def wage_m() -> float:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wage_y",): numpy.array([4.0, 4.0]),
    }
    out = input_data_in_computation_currency(
        input_data__flat=input_data__flat,
        specialized_environment__without_tree_logic_and_with_derived_functions=(
            _spec_env(
                policy_environment={"wage_m": wage_m},
                input_data__flat=input_data__flat,
                grouping_levels=("hh",),
            )
        ),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
        unit_system=UNIT_SYSTEM,
    )
    assert list(numpy.asarray(out[("wage_y",)])) == pytest.approx([1.0, 1.0])


def test_conversion_between_identical_currencies_is_the_identity():
    """Crossing a currency into itself scales nothing: factor 1.0, no qnames."""

    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> float:
        pass

    conversion = CurrencyConversion.between(
        source_currency="CASTAR",
        target_currency="CASTAR",
        qnames=["wealth"],
        specialized_environment={"wealth": wealth},
        unit_system=UNIT_SYSTEM,
    )
    assert conversion.factor == 1.0
    assert conversion.qnames == frozenset()


def test_apply_leaves_object_dtype_column_untouched():
    """An object-dtype column carries `pd.NA` and is handed back as it came in.

    Multiplying it would destroy the missing value the downstream fail-if needs
    to report, so the factor is not applied even though the column's name is one
    the conversion covers.
    """
    conversion = CurrencyConversion(factor=4.0, qnames=frozenset({"wealth"}))
    column = pd.Series([1, pd.NA], dtype="Int64").to_numpy(dtype=object)

    result = conversion.apply(value=column, qname="wealth")

    assert result[0] == 1
    assert result[1] is pd.NA
