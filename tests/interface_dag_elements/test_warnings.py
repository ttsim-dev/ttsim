from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy
import pandas as pd
import pytest

from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.tt import ScalarParam, group_creation_function, policy_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import IntColumn, NestedData, PolicyEnvironment


@pytest.fixture
def minimal_data_tree():
    return {
        "hh_id": numpy.array([1, 2, 3]),
        "p_id": numpy.array([1, 2, 3]),
        "p_id_spouse": numpy.array([2, 1, -1]),
    }


@policy_function()
def some_func(p_id: int) -> int:
    return p_id


@policy_function()
def another_func(some_func: int) -> int:
    return some_func


@pytest.fixture(scope="module")
def minimal_input_data():
    n_individuals = 5
    return {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "fam_id": pd.Series(numpy.arange(n_individuals), name="fam_id"),
    }


def mettsim_environment(backend) -> PolicyEnvironment:
    return main(
        main_target="policy_environment",
        orig_policy_objects={"root": Path(__file__).parent.parent / "mettsim"},
        policy_date=datetime.date(2025, 1, 1),
        backend=backend,
    )


@group_creation_function(
    leaf_name="sp_id", warn_msg_if_included="""You should pass `sp_id` as an input."""
)
def should_warn_sp_id(
    p_id: IntColumn, p_id_spouse: IntColumn, xnp: ModuleType
) -> IntColumn:
    """
    Copy of `sp_id` from METTSIM, but with `warn_msg_if_included` set.
    """
    n = xnp.max(p_id)
    p_id_spouse = xnp.where(p_id_spouse < 0, p_id, p_id_spouse)
    return xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n


@group_creation_function(leaf_name="fam_id")
def dummy_fam_id(sp_id: IntColumn, xnp: ModuleType) -> IntColumn:  # noqa: ARG001
    """
    Just want to use this as a drop-in replacement for `fam_id` from METTSIM with
    minimal inputs.
    """
    return sp_id


def test_warn_if_functions_and_data_columns_overlap(backend):
    with pytest.warns(match="Your data provides the column:"):
        main(
            main_target="warn_if__functions_and_data_columns_overlap",
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "some_func": pd.Series([1]),
                }
            },
            policy_environment={
                "some_func": some_func,
                "another_func": another_func,
            },
            tt_targets={"tree": {"another_func": None}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            include_fail_nodes=False,
            backend=backend,
        )


def test_warn_if_functions_and_columns_overlap_no_warning_if_no_overlap(backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(
            main_target="warn_if__functions_and_data_columns_overlap",
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "x": pd.Series([1]),
                }
            },
            policy_environment={"some_func": some_func},
            tt_targets={"tree": {"some_func": None}},
            policy_date=datetime.date(2025, 1, 1),
            rounding=False,
            include_fail_nodes=False,
            backend=backend,
        )
        assert not w, f"Expected no warning, but got at least: {w[0].message}"


def test_warn_if_evaluation_date_set_in_multiple_places(backend):
    policy_environment = {
        "policy_year": ScalarParam(value=2025),
        "policy_month": ScalarParam(value=1),
        "policy_day": ScalarParam(value=1),
        "evaluation_year": ScalarParam(value=2025),
        "evaluation_month": ScalarParam(value=1),
        "evaluation_day": ScalarParam(value=1),
        "some_func": some_func,
        "another_func": another_func,
    }
    with pytest.warns(match="You have specified the evaluation date in more than one"):
        main(
            main_target="warn_if__evaluation_date_set_in_multiple_places",
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            backend=backend,
        )


def test_warn_if_evaluation_date_set_in_multiple_places_implicitly_added(backend, xnp):
    policy_environment = {
        "policy_year": ScalarParam(value=2025),
        "policy_month": ScalarParam(value=1),
        "policy_day": ScalarParam(value=1),
        "evaluation_year": ScalarParam(value=2025),
        "evaluation_month": ScalarParam(value=1),
        "evaluation_day": ScalarParam(value=1),
        "some_func": some_func,
        "another_func": another_func,
    }
    with pytest.warns(match="You have specified the evaluation date in more than one"):
        main(
            main_target=MainTarget.raw_results.columns,
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            processed_data={"p_id": xnp.array([0])},
            tt_targets=TTTargets(tree={"p_id": None}),
            backend=backend,
        )


def test_do_not_need_to_warn_if_evaluation_date_is_set_only_once(backend, xnp):
    policy_environment = {
        "policy_year": ScalarParam(value=2025),
        "policy_month": ScalarParam(value=1),
        "policy_day": ScalarParam(value=1),
        "some_func": some_func,
        "another_func": another_func,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(
            main_target=MainTarget.raw_results.columns,
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            processed_data={"p_id": xnp.array([0])},
            tt_targets=TTTargets(tree={"p_id": None}),
            backend=backend,
        )
        assert not w, f"Expected no warning, but got at least: {w[0].message}"


def test_warn_if_tt_dag_includes_functions_with_warn_msg_if_included_set(
    minimal_data_tree: NestedData,
    backend: Literal["jax", "numpy"],
):
    env = mettsim_environment(backend)
    env["sp_id"] = should_warn_sp_id
    env["fam_id"] = dummy_fam_id

    with pytest.warns(match="The TT DAG includes elements with `warn_msg_if_included`"):
        main(
            main_target=MainTarget.results.df_with_mapper,
            policy_environment=env,
            tt_targets=TTTargets(tree={"fam_id": None}),
            input_data=InputData.tree(tree=minimal_data_tree),
            backend=backend,
        )
