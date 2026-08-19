from __future__ import annotations

import inspect
from typing import cast

import pytest

from ttsim.interface_dag_elements.automatically_added_functions import (
    _create_function_for_time_unit,
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.time_converters import (
    per_d_to_per_m,
    per_d_to_per_w,
)
from ttsim.tt import TTSIMUnit, policy_function, policy_input
from ttsim.tt.column_objects_param_function import TimeConversionFunction


def return_one() -> int:
    return 1


def return_one_float() -> float:
    return 1.0


def return_true() -> bool:
    return True


def return_x_kin(x_kin: int) -> int:
    return x_kin


def return_n1__x_kin(n1__x_kin: int) -> int:
    return n1__x_kin


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_y", ["test_m", "test_q", "test_w", "test_d"]),
        ("test_y_kin", ["test_m_kin", "test_q_kin", "test_w_kin", "test_d_kin"]),
        ("test_y_sn", ["test_m_sn", "test_q_sn", "test_w_sn", "test_d_sn"]),
        ("test_q", ["test_y", "test_m", "test_w", "test_d"]),
        ("test_q_kin", ["test_y_kin", "test_m_kin", "test_w_kin", "test_d_kin"]),
        ("test_q_sn", ["test_y_sn", "test_m_sn", "test_w_sn", "test_d_sn"]),
        ("test_m", ["test_y", "test_q", "test_w", "test_d"]),
        ("test_m_kin", ["test_y_kin", "test_q_kin", "test_w_kin", "test_d_kin"]),
        ("test_m_sn", ["test_y_sn", "test_q_sn", "test_w_sn", "test_d_sn"]),
        ("test_w", ["test_y", "test_m", "test_q", "test_d"]),
        ("test_w_kin", ["test_y_kin", "test_m_kin", "test_q_kin", "test_d_kin"]),
        ("test_w_sn", ["test_y_sn", "test_m_sn", "test_q_sn", "test_d_sn"]),
        ("test_d", ["test_y", "test_m", "test_q", "test_w"]),
        ("test_d_kin", ["test_y_kin", "test_m_kin", "test_q_kin", "test_w_kin"]),
        ("test_d_sn", ["test_y_sn", "test_m_sn", "test_q_sn", "test_w_sn"]),
    ],
)
def test_should_create_functions_for_other_time_units(
    name: str,
    expected: list[str],
) -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            name: policy_function(leaf_name=name, unit=TTSIMUnit.DIMENSIONLESS)(
                return_one
            ),
        },
        data_qnames=set(),
        grouping_levels=("sn", "kin"),
    )

    for expected_name in expected:
        assert expected_name in time_conversion_functions.all_objects


def test_should_not_create_functions_automatically_that_exist_already() -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            "test1_d": policy_function(
                leaf_name="test1_d", unit=TTSIMUnit.DIMENSIONLESS
            )(return_one),
        },
        data_qnames={"test2_y"},
        grouping_levels=("sn", "kin"),
    )

    assert "test1_d" not in time_conversion_functions.all_objects
    assert "test2_y" not in time_conversion_functions.all_objects


def test_should_overwrite_with_data_cols_differing_only_in_time_period() -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            "test_d": policy_function(leaf_name="test_d", unit=TTSIMUnit.DIMENSIONLESS)(
                return_one
            ),
        },
        data_qnames={"test_y"},
        grouping_levels=("sn", "kin"),
    )

    assert "test_d" in time_conversion_functions.all_objects


def test_create_function_for_time_unit_should_rename_parameter():
    function = _create_function_for_time_unit(source="test", converter=per_d_to_per_m)

    parameter_spec = inspect.getfullargspec(function)
    assert parameter_spec.args == ["test"]


def test_create_function_for_time_unit_should_not_set_info_if_none():
    function = _create_function_for_time_unit(source="test", converter=per_d_to_per_m)

    assert not hasattr(function, "__info__")


def test_create_function_for_time_unit_should_apply_converter(xnp):
    function = _create_function_for_time_unit(source="test", converter=per_d_to_per_w)

    assert function(xnp.array(1)) == 7


def test_time_conversions_should_not_create_cycle():
    # Check for:
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/621
    def x(test_m: int) -> int:
        return test_m

    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            "test_d": policy_function(leaf_name="test_d", unit=TTSIMUnit.DIMENSIONLESS)(
                x
            )
        },
        data_qnames=set(),
        grouping_levels=(),
    )

    assert "test_m" not in time_conversion_functions.all_objects


def test_grouping_functions_should_not_create_cycle():
    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def x(x_hh: int) -> int:
        return x_hh

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def some_other_function_requiring_x_hh(x_hh: int) -> int:
        return x_hh

    grouping_functions = create_agg_by_group_functions(
        column_functions={
            "x": x,
            "some_other_function_requiring_x_hh": some_other_function_requiring_x_hh,
        },
        qname_policy_environment={},
        time_converted_input_stubs={},
        data_qnames=set(),
        tt_targets=("some_other_function_requiring_x_hh",),
        grouping_levels=("hh",),
    )

    assert "x_hh" not in grouping_functions.all_objects


@pytest.mark.parametrize(
    (
        "column_functions",
        "qname_policy_environment",
        "tt_targets",
        "data_qnames",
        "expected",
    ),
    [
        (
            {
                "foo": policy_function(leaf_name="foo", unit=TTSIMUnit.DIMENSIONLESS)(
                    return_x_kin
                )
            },
            {"x": policy_input(unit=TTSIMUnit.DIMENSIONLESS)(return_one)},
            {},
            {"x"},
            ("x_kin"),
        ),
        (
            {
                "n2__foo": policy_function(
                    leaf_name="foo", unit=TTSIMUnit.DIMENSIONLESS
                )(return_n1__x_kin)
            },
            {"n1__x": policy_input(unit=TTSIMUnit.DIMENSIONLESS)(return_one)},
            {},
            {"n1__x"},
            ("n1__x_kin"),
        ),
        (
            {},
            {"x": policy_input(unit=TTSIMUnit.DIMENSIONLESS)(return_one)},
            {"x_kin": None},
            {"x"},
            ("x_kin"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    column_functions,
    qname_policy_environment,
    tt_targets,
    data_qnames,
    expected,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = create_agg_by_group_functions(
        column_functions=column_functions,
        qname_policy_environment=qname_policy_environment,
        time_converted_input_stubs={},
        data_qnames=data_qnames,
        tt_targets=tt_targets,
        grouping_levels=("kin",),
    )
    assert expected in result.all_objects


def test_agg_by_group_resolves_source_dtype_from_sibling_time_unit() -> None:
    """Auto-aggregating a user-supplied input at a different time unit than
    its `PolicyInput` declaration synthesizes the aggregation wrapper by
    resolving the source dtype from the declared sibling.

    `bonus_m` is declared as a `PolicyInput`; the caller supplies `bonus_y`
    via input data; `bonus_y_kin` is requested as a target. The resolver
    walks to the `bonus_m` sibling, reads its declared `data_type`, and
    `create_agg_by_group_functions` produces a typed `bonus_y_kin` wrapper.
    """
    result = create_agg_by_group_functions(
        column_functions={},
        qname_policy_environment={
            "bonus_m": policy_input(unit=TTSIMUnit.DIMENSIONLESS)(return_one_float)
        },
        time_converted_input_stubs={},
        data_qnames={"bonus_y"},
        tt_targets={"bonus_y_kin": None},
        grouping_levels=("kin",),
    )
    assert "bonus_y_kin" in result.all_objects


def test_input_at_a_group_aggregate_name_gets_a_stub_carrying_the_aggregated_unit():
    """Data supplied at `bonus_m_kin` declares the unit a SUM of `bonus_m` implies.

    No aggregation function is created at a name the data supplies — the data would
    override it — so the name would carry no unit for the input-side currency
    conversion and the input-tag checks to read (GEP 10).
    """
    result = create_agg_by_group_functions(
        column_functions={},
        qname_policy_environment={
            "bonus_m": policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)(return_one_float)
        },
        time_converted_input_stubs={},
        data_qnames={"bonus_m_kin"},
        tt_targets={},
        grouping_levels=("kin",),
    )
    assert result.input_stubs["bonus_m_kin"].unit == (
        TTSIMUnit.CURRENCY.PER_MONTH.PER_LEVEL("kin")
    )


def test_input_at_a_group_aggregate_of_a_time_variant_resolves_through_the_sibling():
    """`bonus_y_kin` gets the unit of a SUM over the year-rebased `bonus_m`.

    The aggregation source `bonus_y` is itself a time variant of the declared
    `bonus_m`. A stub resolves its source exactly as a created aggregation function
    does, so a supplied aggregate carries the same unit its generated counterpart
    would.
    """
    result = create_agg_by_group_functions(
        column_functions={},
        qname_policy_environment={
            "bonus_m": policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)(return_one_float)
        },
        time_converted_input_stubs={},
        data_qnames={"bonus_y_kin"},
        tt_targets={},
        grouping_levels=("kin",),
    )
    assert result.input_stubs["bonus_y_kin"].unit == (
        TTSIMUnit.CURRENCY.PER_YEAR.PER_LEVEL("kin")
    )


def test_input_at_a_time_converted_name_gets_a_stub_with_the_rebased_unit():
    """Data supplied at `bonus_y` declares the year-rebased unit of `bonus_m`.

    The supplied name is the source the conversions lead away from, so no function
    is created there and the stub is what carries its unit.
    """
    result = create_time_conversion_functions(
        qname_policy_environment={
            "bonus_m": policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)(return_one_float)
        },
        data_qnames={"bonus_y"},
        grouping_levels=("kin",),
    )
    assert result.input_stubs["bonus_y"].unit == TTSIMUnit.CURRENCY.PER_YEAR


def test_input_at_a_time_variant_of_a_grouped_declaration_is_not_re_derived():
    """`bonus_y_kin` takes its unit from the declared `bonus_m_kin`, not from an
    aggregation of the individual-level `bonus_m`.

    Both derivations reach the name, and a time variant of a declaration at the
    name's own grouping level is the more direct one: aggregating the individual
    column would substitute that column's base and dtype for the grouped
    declaration's.
    """
    result = create_agg_by_group_functions(
        column_functions={},
        qname_policy_environment={
            "bonus_m": policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_MONTH)(return_true)
        },
        time_converted_input_stubs={
            "bonus_y_kin": policy_input(
                unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_LEVEL("kin")
            )(return_one_float)
        },
        data_qnames={"bonus_y_kin"},
        tt_targets={},
        grouping_levels=("kin",),
    )
    assert "bonus_y_kin" not in result.input_stubs


def test_time_conversion_source_ignores_input_at_another_grouping_level():
    """A group-level input column does not feed an individual-level declaration.

    `wage_m_kin` is the household's monthly wage; the individual-level `wage_y`
    must keep converting from itself, not from the group total.
    """
    result = create_time_conversion_functions(
        qname_policy_environment={
            "wage_y": policy_function(leaf_name="wage_y", unit=TTSIMUnit.DIMENSIONLESS)(
                return_one_float
            ),
        },
        data_qnames={"wage_m_kin"},
        grouping_levels=("kin",),
    )

    source = cast("TimeConversionFunction", result.all_objects["wage_d"]).source
    assert source == "wage_y"


def test_time_conversion_source_ignores_ungrouped_input_for_grouped_declaration():
    """An individual-level input column does not feed a group-level declaration.

    `wage_m` is a person's monthly wage; the group-level `wage_y_kin` must keep
    converting from itself, not from the per-person amount.
    """
    result = create_time_conversion_functions(
        qname_policy_environment={
            "wage_y_kin": policy_function(
                leaf_name="wage_y_kin", unit=TTSIMUnit.DIMENSIONLESS
            )(return_one_float),
        },
        data_qnames={"wage_m"},
        grouping_levels=("kin",),
    )

    source = cast("TimeConversionFunction", result.all_objects["wage_d_kin"]).source
    assert source == "wage_y_kin"


def test_time_conversion_source_is_the_input_at_the_same_grouping_level():
    """A same-level input column is the source the conversions lead away from."""
    result = create_time_conversion_functions(
        qname_policy_environment={
            "wage_y_kin": policy_function(
                leaf_name="wage_y_kin", unit=TTSIMUnit.DIMENSIONLESS
            )(return_one_float),
        },
        data_qnames={"wage_m_kin"},
        grouping_levels=("kin",),
    )

    source = cast("TimeConversionFunction", result.all_objects["wage_d_kin"]).source
    assert source == "wage_m_kin"
