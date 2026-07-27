from __future__ import annotations

import datetime

import dags.tree as dt
import numpy
import pytest

from tests.test_unit_system import TEST_UNIT_SYSTEM
from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.interface_dag_elements.currency import (
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
        data_qnames={dt.qname_from_tree_path(p) for p in input_data__flat},
        grouping_levels=grouping_levels,
    )


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
        unit_system=TEST_UNIT_SYSTEM,
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
        unit_system=TEST_UNIT_SYSTEM,
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
        unit_system=TEST_UNIT_SYSTEM,
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
        unit_system=TEST_UNIT_SYSTEM,
    )
    assert list(numpy.asarray(out[("wage_y",)])) == pytest.approx([1.0, 1.0])


def test_object_dtype_input_fails_before_currency_conversion():
    """Input validation rejects object arrays before currency arithmetic."""

    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> float:
        pass

    with pytest.raises(ValueError, match=r"(?s)object dtype.*wealth"):
        main(
            main_target=MainTarget.processed_data,
            input_data=InputData.tree(
                {
                    "p_id": numpy.array([0, 1]),
                    "wealth": numpy.array(["unknown", "unknown"], dtype=object),
                }
            ),
            policy_environment={"wealth": wealth},
            policy_date=datetime.date(2020, 1, 1),
            tt_targets=TTTargets.tree({"wealth": None}),
            data_currency="SILVER_PENNY",
            unit_system=TEST_UNIT_SYSTEM,
        )


def test_identity_currency_conversion_preserves_integer_dtype():
    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> int:
        pass

    column = numpy.array([1, 2], dtype=numpy.int32)
    result = input_data_in_computation_currency(
        input_data__flat={
            ("p_id",): numpy.array([0, 1]),
            ("wealth",): column,
        },
        specialized_environment__without_tree_logic_and_with_derived_functions={
            "wealth": wealth
        },
        data_currency="CASTAR",
        computation_currency="CASTAR",
        unit_system=TEST_UNIT_SYSTEM,
    )

    assert result[("wealth",)].dtype == numpy.dtype("int32")
