from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim import middle_earth

from ttsim import InputData, MainTarget, OrigPolicyObjects, TTTargets, main
from ttsim.tt import AggType, Unit, agg_by_group_function
from ttsim.tt.column_objects_param_function import policy_function, policy_input
from ttsim.typing import FloatColumn

if TYPE_CHECKING:
    from typing import Literal


DF_WITH_NESTED_COLUMNS = pd.DataFrame(
    {
        ("age",): [10, 30, 30],
        ("kin_id",): [0, 0, 0],
        ("p_id",): [2, 0, 1],
        ("p_id_parent_1",): [0, -1, -1],
        ("p_id_parent_2",): [1, -1, -1],
        ("p_id_spouse",): [-1, 1, 0],
        ("parent_is_noble",): [False, False, False],
        ("wealth",): [0.0, 0.0, 0.0],
        ("payroll_tax", "child_tax_credit", "p_id_recipient"): [0, -1, -1],
        ("payroll_tax", "income", "gross_wage_y"): [0, 10000, 0],
    },
)


DF_WITH_QNAME_COLUMNS = pd.DataFrame(
    {
        "age": [10, 30, 30],
        "kin_id": [0, 0, 0],
        "p_id": [2, 0, 1],
        "p_id_parent_1": [0, -1, -1],
        "p_id_parent_2": [1, -1, -1],
        "p_id_spouse": [-1, 1, 0],
        "parent_is_noble": [False, False, False],
        "wealth": [0.0, 0.0, 0.0],
        "payroll_tax__child_tax_credit__p_id_recipient": [0, -1, -1],
        "payroll_tax__income__gross_wage_y": [0, 10000, 0],
    },
)


DF_FOR_MAPPER = pd.DataFrame(
    {
        "age": [10, 30, 30],
        "kin_id": [0, 0, 0],
        "p_id": [2, 0, 1],
        "parent_1": [0, -1, -1],
        "parent_2": [1, -1, -1],
        "spouse": [-1, 1, 0],
        "parent_is_noble": [False, False, False],
        "child_tax_credit_recipient": [0, -1, -1],
        "gross_wage_y": [0, 10000, 0],
        "wealth": [0.0, 0.0, 0.0],
    },
)


INPUT_QNAME_DATA = {
    "age": numpy.array([10, 30, 30]),
    "kin_id": numpy.array([0, 0, 0]),
    "p_id": numpy.array([2, 0, 1]),
    "p_id_parent_1": numpy.array([0, -1, -1]),
    "p_id_parent_2": numpy.array([1, -1, -1]),
    "p_id_spouse": numpy.array([-1, 1, 0]),
    "parent_is_noble": numpy.array([False, False, False]),
    "wealth": numpy.array([0.0, 0.0, 0.0]),
    "payroll_tax__child_tax_credit__p_id_recipient": numpy.array([0, -1, -1]),
    "payroll_tax__income__gross_wage_y": numpy.array([0, 10000, 0]),
}


INPUT_DF_MAPPER = {
    "age": "age",
    "kin_id": "kin_id",
    "p_id": "p_id",
    "p_id_parent_1": "parent_1",
    "p_id_parent_2": "parent_2",
    "p_id_spouse": "spouse",
    "parent_is_noble": "parent_is_noble",
    "wealth": "wealth",
    "payroll_tax": {
        "child_tax_credit": {
            "p_id_recipient": "child_tax_credit_recipient",
        },
        "income": {
            "gross_wage_y": "gross_wage_y",
        },
    },
}


INPUT_TREE_DATA = {
    "age": numpy.array([10, 30, 30]),
    "kin_id": numpy.array([0, 0, 0]),
    "p_id": numpy.array([2, 0, 1]),
    "p_id_parent_1": numpy.array([0, -1, -1]),
    "p_id_parent_2": numpy.array([1, -1, -1]),
    "p_id_spouse": numpy.array([-1, 1, 0]),
    "parent_is_noble": numpy.array([False, False, False]),
    "wealth": numpy.array([0.0, 0.0, 0.0]),
    "payroll_tax": {
        "child_tax_credit": {"p_id_recipient": numpy.array([0, -1, -1])},
        "income": {"gross_wage_y": numpy.array([0, 10000, 0])},
    },
}


INPUT_FLAT_DATA = {
    ("age",): numpy.array([10, 30, 30]),
    ("kin_id",): numpy.array([0, 0, 0]),
    ("p_id",): numpy.array([2, 0, 1]),
    ("p_id_parent_1",): numpy.array([0, -1, -1]),
    ("p_id_parent_2",): numpy.array([1, -1, -1]),
    ("p_id_spouse",): numpy.array([-1, 1, 0]),
    ("parent_is_noble",): numpy.array([False, False, False]),
    ("wealth",): numpy.array([0.0, 0.0, 0.0]),
    ("payroll_tax", "child_tax_credit", "p_id_recipient"): numpy.array([0, -1, -1]),
    ("payroll_tax", "income", "gross_wage_y"): numpy.array([0, 10000, 0]),
}


TARGETS_TREE = {
    "payroll_tax": {
        "amount_y": "payroll_tax_amount_y",
        "child_tax_credit": {
            "amount_m": "payroll_tax_child_tax_credit_amount_m",
        },
    },
}


TARGETS_TREE_NO_RENAME = {
    "payroll_tax": {
        "amount_y": None,
        "child_tax_credit": {"amount_m": None},
    },
}


EXPECTED_TT_RESULTS = pd.DataFrame(
    {
        "payroll_tax_amount_y": [0.0, 2980.0, 0.0],
        "payroll_tax_child_tax_credit_amount_m": [0.0, 2.083333, 0.0],
    },
    index=pd.Index([2, 0, 1], name="p_id"),
)


@policy_input(unit=Unit.DIMENSIONLESS)
def p_id() -> int:
    pass


@policy_input(unit=Unit.CURRENCY_FLOW)
def income_m() -> float:
    pass


@policy_function(vectorization_strategy="vectorize", unit=Unit.CURRENCY_FLOW)
def benefit_m(income_m: float) -> float:
    return income_m * 0.5


@pytest.mark.parametrize(
    "input_data_arg",
    [
        InputData.df_and_mapper(df=DF_FOR_MAPPER, mapper=INPUT_DF_MAPPER),
        InputData.df_with_nested_columns(DF_WITH_NESTED_COLUMNS),
        InputData.df_with_qname_columns(DF_WITH_QNAME_COLUMNS),
        InputData.tree(INPUT_TREE_DATA),
        InputData.flat(INPUT_FLAT_DATA),
        InputData.qname(INPUT_QNAME_DATA),
    ],
    ids=[
        "df_and_mapper",
        "df_with_nested_columns",
        "df_with_qname_columns",
        "tree",
        "flat",
        "qname",
    ],
)
def test_end_to_end(input_data_arg, backend: Literal["numpy", "jax"]):
    """Every `InputData.*` shape produces the same `Results.df_with_mapper`."""
    result = main(
        main_target=(MainTarget.results.df_with_mapper),
        input_data=input_data_arg,
        tt_targets=TTTargets.tree(TARGETS_TREE),
        policy_date_str="2025-01-01",
        rounding=False,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
    )
    pd.testing.assert_frame_equal(
        EXPECTED_TT_RESULTS,
        result,
        check_dtype=False,
        check_index_type=False,
    )


_EXPECTED_AMOUNTS_IN_USER_ORDER = numpy.array([0.0, 2980.0, 0.0])
_EXPECTED_CHILD_TAX_CREDIT_IN_USER_ORDER = numpy.array([0.0, 2.083333, 0.0])
_USER_P_IDS_IN_ORDER = numpy.array([2, 0, 1])


@pytest.mark.parametrize(
    ("output_target", "shape_id"),
    [
        (MainTarget.results.tree, "tree"),
        (MainTarget.results.flat, "flat"),
        (MainTarget.results.qname, "qname"),
        (MainTarget.results.df_with_nested_columns, "df_with_nested_columns"),
        (MainTarget.results.df_with_qname_columns, "df_with_qname_columns"),
    ],
    ids=["tree", "flat", "qname", "df_with_nested_columns", "df_with_qname_columns"],
)
def test_results_shapes_render_payroll_tax_amount_y(
    output_target: str, shape_id: str, backend: Literal["numpy", "jax"]
):
    """Every `Results.*` output shape exposes `payroll_tax__amount_y` with the
    same per-row values, indexed in the user's original p_id order.
    """
    result = main(
        main_target=output_target,
        input_data=InputData.df_with_nested_columns(DF_WITH_NESTED_COLUMNS),
        tt_targets=TTTargets.tree(TARGETS_TREE_NO_RENAME),
        policy_date_str="2025-01-01",
        rounding=False,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
    )

    if shape_id == "tree":
        amounts = numpy.asarray(result["payroll_tax"]["amount_y"])
    elif shape_id == "flat":
        amounts = numpy.asarray(result[("payroll_tax", "amount_y")])
    elif shape_id == "qname":
        amounts = numpy.asarray(result["payroll_tax__amount_y"])
    elif shape_id == "df_with_nested_columns":
        # The MultiIndex pads shorter paths with NaN; lookup returns a 2-D
        # frame whose single column carries the values for this leaf.
        amounts = result.xs("amount_y", level=1, axis="columns").to_numpy().squeeze()
        assert list(result.index) == list(_USER_P_IDS_IN_ORDER)
    else:  # df_with_qname_columns
        amounts = result["payroll_tax__amount_y"].to_numpy()
        assert list(result.index) == list(_USER_P_IDS_IN_ORDER)

    numpy.testing.assert_allclose(amounts, _EXPECTED_AMOUNTS_IN_USER_ORDER)


@pytest.mark.parametrize(
    "wage_dtype",
    ["uint32", "UInt32", "uint32[pyarrow]"],
)
def test_uint_wage_input_does_not_underflow(
    wage_dtype: str, backend: Literal["numpy", "jax"]
):
    """A `uint`-typed wage column flows through `max(gross - deductions, 0)` as
    signed arithmetic, so the result is 0 rather than the uint wraparound.
    """
    if wage_dtype.endswith("[pyarrow]"):
        pytest.importorskip("pyarrow")
    nested_columns_df = DF_WITH_NESTED_COLUMNS.copy()
    nested_columns_df[("payroll_tax", "income", "gross_wage_y")] = pd.Series(
        [0, 0, 0], dtype=wage_dtype
    )
    result = main(
        main_target=MainTarget.results.tree,
        input_data=InputData.df_with_nested_columns(nested_columns_df),
        tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
        policy_date_str="2025-01-01",
        rounding=False,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
    )
    amount_y = result["payroll_tax"]["amount_y"]
    assert float(amount_y[0]) == 0.0
    assert float(amount_y[1]) == 0.0
    assert float(amount_y[2]) == 0.0


def test_df_with_qname_columns_has_qname_string_columns(
    backend: Literal["numpy", "jax"],
):
    """Flat qname-named columns survive adding/removing targets without changing
    the column index depth — unlike `df_with_nested_columns` which uses a
    MultiIndex whose depth tracks the deepest target.
    """
    result = main(
        main_target=MainTarget.results.df_with_qname_columns,
        input_data=InputData.df_with_nested_columns(DF_WITH_NESTED_COLUMNS),
        tt_targets=TTTargets.tree(
            {
                "payroll_tax": {
                    "amount_y": None,
                    "child_tax_credit": {"amount_m": None},
                },
            }
        ),
        policy_date_str="2025-01-01",
        rounding=False,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
    )
    assert list(result.columns) == [
        "payroll_tax__amount_y",
        "payroll_tax__child_tax_credit__amount_m",
    ]
    assert result.index.name == "p_id"
    assert list(result.index) == [2, 0, 1]
    pd.testing.assert_series_equal(
        result["payroll_tax__amount_y"].reset_index(drop=True),
        EXPECTED_TT_RESULTS["payroll_tax_amount_y"].reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["payroll_tax__child_tax_credit__amount_m"].reset_index(drop=True),
        EXPECTED_TT_RESULTS["payroll_tax_child_tax_credit_amount_m"].reset_index(
            drop=True
        ),
        check_dtype=False,
        check_names=False,
    )


def _run_cloudpickle_subprocess(tmp_path, script_body: str) -> None:
    """Run a cloudpickle round-trip in a fresh subprocess.

    A subprocess avoids pytest's stdout-capture wrappers leaking into
    module-level closures (which would otherwise taint cloudpickle).
    """
    pytest.importorskip("cloudpickle")
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    script = tmp_path / "repro.py"
    script.write_text(textwrap.dedent(script_body))
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_cloudpickle_round_trip_preserves_tt_function_output(tmp_path):
    """A `tt_function` built against a policy environment loaded from a
    ROOT_PATH (here: mettsim) survives a cloudpickle round-trip with results
    unchanged on every requested target.
    """
    _run_cloudpickle_subprocess(
        tmp_path=tmp_path,
        script_body="""
        import cloudpickle
        import numpy as np
        from mettsim import middle_earth

        from ttsim import InputData, OrigPolicyObjects, TTTargets, main

        data = {
            ("age",): np.array([30, 30]),
            ("kin_id",): np.array([0, 0]),
            ("p_id",): np.array([0, 1]),
            ("p_id_parent_1",): np.array([-1, -1]),
            ("p_id_parent_2",): np.array([-1, -1]),
            ("p_id_spouse",): np.array([1, 0]),
            ("parent_is_noble",): np.array([False, False]),
            ("payroll_tax", "child_tax_credit", "p_id_recipient"):
                np.array([-1, -1]),
            ("payroll_tax", "income", "gross_wage_y"):
                np.array([10000.0, 0.0]),
            ("wealth",): np.array([0.0, 0.0]),
        }
        kwargs = dict(
            policy_date_str="2025-01-01",
            input_data=InputData.flat(data),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
            backend="numpy",
        )
        tt_func = main(main_target="tt_function", **kwargs)
        processed = main(main_target="processed_data", **kwargs)
        restored = cloudpickle.loads(cloudpickle.dumps(tt_func))
        expected = tt_func(processed)
        actual = restored(processed)
        for qname in expected:
            np.testing.assert_array_equal(expected[qname], actual[qname])
        print("OK")
        """,
    )


def test_cloudpickle_round_trip_with_inline_policy_environment(tmp_path):
    """A `tt_function` built from a user-defined inline `policy_environment`
    (no `OrigPolicyObjects.root`, no on-disk policy package) also survives a
    cloudpickle round-trip with results unchanged.
    """
    _run_cloudpickle_subprocess(
        tmp_path=tmp_path,
        script_body="""
        import datetime
        import cloudpickle
        import numpy as np

        from ttsim import InputData, TTTargets, main
        from ttsim.tt import Unit, policy_function, policy_input


        @policy_input(unit=Unit.DIMENSIONLESS)
        def p_id() -> int: ...


        @policy_input(unit=Unit.CURRENCY_FLOW)
        def income_m() -> float: ...


        @policy_function(vectorization_strategy="vectorize", unit=Unit.CURRENCY_FLOW)
        def benefit_m(income_m: float) -> float:
            return income_m * 0.5


        env = {"p_id": p_id, "income_m": income_m, "benefit_m": benefit_m}
        kwargs = dict(
            policy_environment=env,
            input_data=InputData.tree({
                "p_id": np.array([0, 1, 2]),
                "income_m": np.array([1000.0, 2000.0, 3000.0]),
            }),
            tt_targets=TTTargets.tree({"benefit_m": None}),
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend="numpy",
        )
        tt_func = main(main_target="tt_function", **kwargs)
        processed = main(main_target="processed_data", **kwargs)
        root_nodes = main(main_target="labels__root_nodes", **kwargs)
        filtered = {k: v for k, v in processed.items() if k in root_nodes}

        restored = cloudpickle.loads(cloudpickle.dumps(tt_func))
        expected = tt_func(filtered)
        actual = restored(filtered)
        for qname in expected:
            np.testing.assert_array_equal(expected[qname], actual[qname])
        print("OK")
        """,
    )


def test_can_create_input_template(backend: Literal["numpy", "jax"]):
    result_template = main(
        main_target=MainTarget.templates.input_data_dtypes.tree,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
        tt_targets=TTTargets.tree(TARGETS_TREE),
    )
    flat_result_template = dt.flatten_to_tree_paths(result_template)
    flat_expected = dt.flatten_to_tree_paths(INPUT_DF_MAPPER)
    assert flat_result_template.keys() == flat_expected.keys()


def test_modify_evaluation_date_after_creating_policy_environment(
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
):
    policy_environment = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2000-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
    )
    input_data = InputData.tree(
        tree={
            "p_id": xnp.array([2, 0, 1]),
            "property_tax": {
                "acre_size": xnp.array([200, 5, 20]),
            },
        }
    )
    result = main(
        main_target=MainTarget.results.df_with_mapper,
        policy_environment=policy_environment,
        # acre_size capped starting in 2020
        evaluation_date_str="2020-01-01",
        input_data=input_data,
        tt_targets=TTTargets.tree(
            {"property_tax": {"amount_y": "property_tax_amount_y"}}
        ),
        backend=backend,
    )
    expected = pd.DataFrame(
        {
            # The 1900 schedule is denominated in silver pennies; a default
            # (castar) run converts it at build: 1000 pennies = 250 castar.
            "property_tax_amount_y": [250.0, 0.0, 250.0],
        },
        index=pd.Index([2, 0, 1], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_different_evaluation_dates_across_data_rows(
    backend: Literal["numpy", "jax"], xnp: ModuleType
):
    @policy_function(unit=Unit.YEARS)
    def f(evaluation_year: int) -> int:
        return evaluation_year

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "f": f,
        },
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([3, 1, 2]),
                "evaluation_year": xnp.array([2022, 2020, 2021]),
            }
        ),
        tt_targets=TTTargets.tree({"f": None}),
        backend=backend,
    )

    expected = pd.DataFrame(
        {
            ("f",): [2022, 2020, 2021],
        },
        index=pd.Index([3, 1, 2], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_input_data_as_targets(xnp: ModuleType, backend: Literal["numpy", "jax"]):
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_date_str="2025-01-01",
        input_data=InputData.tree(
            {
                "kin_id": xnp.array([0, 0, 0]),
                "payroll_tax": {
                    "amount_y": xnp.array([0, 1000, 0]),
                },
                "p_id": xnp.array([2, 0, 1]),
            }
        ),
        tt_targets=TTTargets.tree({"kin_id": None, "payroll_tax": {"amount_y": None}}),
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
        include_warn_nodes=False,
    )
    expected = pd.DataFrame(
        {
            ("kin_id",): [0, 0, 0],
            ("payroll_tax", "amount_y"): [0, 1000, 0],
        },
        index=pd.Index([2, 0, 1], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_input_data_reordering_with_distinct_values(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """Test that demonstrates input data gets reordered internally and restored
    correctly.
    """
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_date_str="2025-01-01",
        input_data=InputData.tree(
            {
                "age": xnp.array([25, 45, 35]),
                "wealth": xnp.array([1000, 2000, 3000]),
                "p_id": xnp.array([2, 0, 1]),
            }
        ),
        # Request input columns as outputs to see if they maintain correct order
        tt_targets=TTTargets.tree({"age": None, "wealth": None}),
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
        include_warn_nodes=False,
    )

    # Expected: Values should appear in same positions as original p_id order
    expected = pd.DataFrame(
        {
            ("age",): [25, 45, 35],
            ("wealth",): [1000, 2000, 3000],
        },
        index=pd.Index([2, 0, 1], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_derived_time_converted_scalar_can_partialled(xnp, backend):
    """Scalar inputs are partialled correctly.

    Scalar inputs are partialled also if they replace a function that is derived from
    a policy input.
    """
    policy_environment = {
        "p_id": p_id,
        "income_m": income_m,
        "benefit_m": benefit_m,
    }
    input_data = {
        "p_id": xnp.array([1, 2, 3]),
        "income_y": 12000,
    }
    root_nodes = main(
        main_target=MainTarget.labels.root_nodes,
        policy_environment=policy_environment,
        input_data=InputData.tree(input_data),
        tt_targets=TTTargets.tree({"benefit_m": None}),
        policy_date_str="2024-01-01",
        evaluation_date_str="2024-01-01",
        backend=backend,
        include_warn_nodes=False,
    )
    assert root_nodes == set()


@policy_input(unit=Unit.DIMENSIONLESS)
def broadcast_x() -> float:
    pass


@policy_function(vectorization_strategy="not_required", unit=Unit.DIMENSIONLESS)
def cumulative_broadcast_x(broadcast_x: FloatColumn, xnp: ModuleType) -> FloatColumn:
    """Declared with vectorization_strategy='not_required'; it operates on the whole
    array and cannot run on a bare scalar.
    """
    return xnp.cumsum(broadcast_x)


@agg_by_group_function(agg_type=AggType.SUM, unit=Unit.DIMENSIONLESS)
def broadcast_x_fam(broadcast_x: float, fam_id: int) -> float:
    pass


@policy_input(unit=Unit.DIMENSIONLESS)
def fam_id() -> int:
    pass


def test_scalar_input_to_not_required_function_is_broadcast(xnp, backend):
    """A scalar bound to a `Column` argument of a function with
    vectorization_strategy='not_required' is broadcast to the population length at call
    time, so the function sees a full-length array.
    """
    results = main(
        main_target=MainTarget.results.tree,
        policy_environment={
            "p_id": p_id,
            "broadcast_x": broadcast_x,
            "cumulative_broadcast_x": cumulative_broadcast_x,
        },
        input_data=InputData.tree(
            {"p_id": xnp.array([1, 2, 3]), "broadcast_x": 100.0},
        ),
        tt_targets=TTTargets.tree({"cumulative_broadcast_x": None}),
        policy_date_str="2024-01-01",
        evaluation_date_str="2024-01-01",
        backend=backend,
        include_warn_nodes=False,
    )
    numpy.testing.assert_array_equal(
        results["cumulative_broadcast_x"],
        numpy.array([100.0, 200.0, 300.0]),
    )


def test_scalar_input_to_not_required_function_is_baked_in(xnp, backend):
    """The broadcast scalar is partialled in, so it is not a root node. The population
    length is taken from the partialled-in `len_p_id`, so the broadcast introduces no
    root node of its own and none remain.
    """
    root_nodes = main(
        main_target=MainTarget.labels.root_nodes,
        policy_environment={
            "p_id": p_id,
            "broadcast_x": broadcast_x,
            "cumulative_broadcast_x": cumulative_broadcast_x,
        },
        input_data=InputData.tree(
            {"p_id": xnp.array([1, 2, 3]), "broadcast_x": 100.0},
        ),
        tt_targets=TTTargets.tree({"cumulative_broadcast_x": None}),
        policy_date_str="2024-01-01",
        evaluation_date_str="2024-01-01",
        backend=backend,
        include_warn_nodes=False,
    )
    assert "broadcast_x" not in root_nodes
    assert root_nodes == set()


@policy_input(unit=Unit.CURRENCY_FLOW)
def bonus_m() -> float:
    pass


@policy_function(vectorization_strategy="vectorize", unit=Unit.CURRENCY_FLOW)
def doubled_y_fam(bonus_y_fam: float) -> float:
    return 2.0 * bonus_y_fam


def test_auto_aggregation_resolves_dtype_from_sibling_time_unit(
    backend: Literal["numpy", "jax"], xnp: ModuleType
):
    """Auto-aggregating an input supplied at a different time unit than its
    `PolicyInput` declaration succeeds by resolving the source dtype from
    the declared sibling.

    `bonus_m` is declared as a `PolicyInput`; the caller supplies `bonus_y`
    in input data; `doubled_y_fam` consumes the `bonus_y_fam` auto-aggregation.
    The resolver walks to `bonus_m` for the dtype (and, GEP 10, for the
    non-time unit), the SUM-by-`fam` wrapper is synthesised, and
    `doubled_y_fam` returns twice the per-`fam` yearly sum.
    """
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "fam_id": fam_id,
            "bonus_m": bonus_m,
            "doubled_y_fam": doubled_y_fam,
        },
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([0, 1, 2]),
                "fam_id": xnp.array([0, 0, 1]),
                "bonus_y": xnp.array([1200.0, 600.0, 2400.0]),
            },
        ),
        tt_targets=TTTargets.tree({"doubled_y_fam": None}),
        policy_date_str="2025-01-01",
        evaluation_date_str="2025-01-01",
        rounding=False,
        backend=backend,
        include_warn_nodes=False,
    )
    expected = pd.DataFrame(
        {("doubled_y_fam",): [3600.0, 3600.0, 4800.0]},
        index=pd.Index([0, 1, 2], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )
