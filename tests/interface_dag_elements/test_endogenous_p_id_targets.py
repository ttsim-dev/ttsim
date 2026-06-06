"""Tests for endogenously-computed `p_id_*` columns as targets.

When a policy function computes a `p_id_*` column (e.g. `p_id_recipient`),
its output values are *internal* indices into the sorted-by-p_id array. The
interface DAG must reverse-translate those internal indices back to
user-space `p_id` values before returning results.
"""

from __future__ import annotations

import datetime
import inspect
from typing import TYPE_CHECKING

import pandas as pd
from dags import get_free_arguments

from ttsim import (
    InputData,
    MainTarget,
    SpecializedEnvironment,
    TTTargets,
    main,
)
from ttsim.tt import FKType, ScalarParam, policy_function, policy_input

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal


_DATE = datetime.date(2025, 1, 1)


def _policy_year_month_day() -> dict[str, ScalarParam]:
    return {
        "policy_year": ScalarParam(value=_DATE.year, start_date=_DATE, end_date=_DATE),
        "policy_month": ScalarParam(
            value=_DATE.month, start_date=_DATE, end_date=_DATE
        ),
        "policy_day": ScalarParam(value=_DATE.day, start_date=_DATE, end_date=_DATE),
    }


def _identity(p_id: int) -> int:
    return p_id


def _recipient_or_minus_one(p_id: int, is_recipient: bool) -> int:
    return p_id if is_recipient else -1


def test_endogenous_p_id_target_returns_user_space_p_ids(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """`p_id_*` computed by the policy is reverse-translated to user-space p_ids.

    `p_id_recipient = p_id` should yield each row's own user-space `p_id` —
    not the internal sorted index — even when the input `p_id` order is
    non-monotonic.
    """
    p_id_recipient = policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="p_id_recipient",
    )(_identity)

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "p_id_recipient": p_id_recipient,
            **_policy_year_month_day(),
        },
        tt_targets=TTTargets.tree({"p_id_recipient": None}),
        input_data=InputData.tree(tree={"p_id": xnp.array([20, 10, 30])}),
        backend=backend,
    )

    expected = pd.DataFrame(
        {("p_id_recipient",): [20, 10, 30]},
        index=pd.Index([20, 10, 30], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def _recipient_or_negative_two(p_id: int, is_recipient: bool) -> int:
    return p_id if is_recipient else -2


def test_endogenous_p_id_target_collapses_arbitrary_negative_to_sentinel(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """Any negative integer a policy function returns — not just `-1` — maps
    to the no-link sentinel `-1` in the user-space output. Treating only
    `-1` as the sentinel would silently misroute other negatives (the
    `xnp.maximum(col, 0)` clamp would gather `sorted_orig_p_ids[0]`).
    """
    p_id_recipient = policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="p_id_recipient",
        vectorization_strategy="vectorize",
    )(_recipient_or_negative_two)

    @policy_input()
    def is_recipient() -> bool:
        pass

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "p_id_recipient": p_id_recipient,
            "is_recipient": is_recipient,
            **_policy_year_month_day(),
        },
        tt_targets=TTTargets.tree({"p_id_recipient": None}),
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([20, 10, 30]),
                "is_recipient": xnp.array([True, False, True]),
            }
        ),
        backend=backend,
    )

    expected = pd.DataFrame(
        {("p_id_recipient",): [20, -1, 30]},
        index=pd.Index([20, 10, 30], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_endogenous_p_id_target_preserves_minus_one_sentinel(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """A `-1` (no-link) sentinel from the policy survives the reverse-translation
    rather than being mapped to whichever user-space `p_id` happens to sit at
    sorted index 0.
    """
    p_id_recipient = policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="p_id_recipient",
        vectorization_strategy="vectorize",
    )(_recipient_or_minus_one)

    @policy_input()
    def is_recipient() -> bool:
        pass

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "p_id_recipient": p_id_recipient,
            "is_recipient": is_recipient,
            **_policy_year_month_day(),
        },
        tt_targets=TTTargets.tree({"p_id_recipient": None}),
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([20, 10, 30]),
                "is_recipient": xnp.array([True, False, True]),
            }
        ),
        backend=backend,
    )

    expected = pd.DataFrame(
        {("p_id_recipient",): [20, -1, 30]},
        index=pd.Index([20, 10, 30], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_exogenous_p_id_pointer_as_input_target_is_unchanged(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """An exogenous `p_id_parent_1` column requested as a target is delivered
    verbatim — values stay in user-space `p_id`, including the `-1` sentinel,
    because it flows through `raw_results.from_input_data` rather than the
    computed-column path.
    """

    @policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF)
    def p_id_parent_1() -> int:
        pass

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "p_id_parent_1": p_id_parent_1,
            **_policy_year_month_day(),
        },
        tt_targets=TTTargets.tree({"p_id_parent_1": None}),
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([20, 10, 30]),
                "p_id_parent_1": xnp.array([30, -1, -1]),
            }
        ),
        backend=backend,
    )

    expected = pd.DataFrame(
        {("p_id_parent_1",): [30, -1, -1]},
        index=pd.Index([20, 10, 30], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_endogenous_p_id_target_mixed_with_regular_column(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """Endogenous `p_id_*` and a regular numeric column requested in one call:
    the former is reverse-translated, the latter passes through untouched.
    """
    p_id_recipient = policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="p_id_recipient",
    )(_identity)

    @policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="doubled",
        vectorization_strategy="vectorize",
    )
    def doubled(income_m: float) -> float:
        return income_m * 2.0

    @policy_input()
    def income_m() -> float:
        pass

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "p_id_recipient": p_id_recipient,
            "doubled": doubled,
            "income_m": income_m,
            **_policy_year_month_day(),
        },
        tt_targets=TTTargets.tree({"p_id_recipient": None, "doubled": None}),
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([20, 10, 30]),
                "income_m": xnp.array([100.0, 200.0, 300.0]),
            }
        ),
        backend=backend,
    )

    expected = pd.DataFrame(
        {
            ("p_id_recipient",): [20, 10, 30],
            ("doubled",): [200.0, 400.0, 600.0],
        },
        index=pd.Index([20, 10, 30], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected,
        result,
        check_dtype=False,
        check_index_type=False,
        check_like=True,
    )


def test_jittable_with_specialized_environment_and_dummy_processed_data(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    """Jit-compile a single policy function from dummy `processed_data` and a
    pre-built specialized environment, without any `input_data` (regression
    test for #130). This mirrors GETTSIM's `test_jittable`: the dummy data is
    derived from the function's free arguments, and the target is
    `raw_results.columns_with_internal_p_ids` so that the remapping to
    user-space `p_id`s (`raw_results.columns_with_original_p_ids`) is pruned.
    """
    p_id_recipient = policy_function(
        start_date=_DATE,
        end_date=_DATE,
        leaf_name="p_id_recipient",
    )(_identity)

    full_env = main(
        main_target=(
            "specialized_environment_for_plotting_and_templates",
            "with_partialled_params_and_scalars",
        ),
        policy_environment={
            "p_id_recipient": p_id_recipient,
            **_policy_year_month_day(),
        },
        backend=backend,
        include_fail_nodes=False,
        include_warn_nodes=False,
    )
    env = {"p_id_recipient": full_env["p_id_recipient"]}

    processed_data = {}
    for arg_name in get_free_arguments(env["p_id_recipient"]):
        annotation = str(
            inspect.signature(env["p_id_recipient"]).parameters[arg_name].annotation
        )
        if "FloatColumn" in annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=float)
        elif "IntColumn" in annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=int)
        elif "BoolColumn" in annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=bool)
        else:
            raise ValueError(f"Unknown column type: {annotation}")

    result = main(
        main_target=MainTarget.raw_results.columns_with_internal_p_ids,
        specialized_environment=(
            SpecializedEnvironment.with_partialled_params_and_scalars(env)
        ),
        processed_data=processed_data,
        tt_targets=TTTargets.qname(["p_id_recipient"]),
        backend=backend,
        include_fail_nodes=False,
        include_warn_nodes=False,
    )

    # Raw results stay in internal representation: no remapping happens.
    assert list(result["p_id_recipient"]) == [0]
