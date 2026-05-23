from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttsim import InputData, TTTargets, main


def test_scalar_leaf_in_tree_is_broadcast_to_p_id_length():
    flat = main(
        main_target="input_data__flat",
        input_data=InputData.tree(
            {
                "p_id": np.array([0, 1, 2]),
                "x": 5.0,
            }
        ),
        tt_targets=TTTargets.tree({"x": None}),
        backend="numpy",
    )
    x = flat[("x",)]
    assert len(x) == 3
    assert np.all(x == 5.0)


@pytest.mark.parametrize(
    "input_data_factory",
    [
        lambda: InputData.tree({"p_id": np.array([0, 1, 2]), "x": 5.0}),
        lambda: InputData.df_and_mapper(
            df=pd.DataFrame({"pid_col": [0, 1, 2]}),
            mapper={"p_id": "pid_col", "x": 5.0},
        ),
        lambda: InputData.df_with_nested_columns(
            pd.DataFrame({("p_id",): [0, 1, 2], ("x",): [5.0, 5.0, 5.0]})
        ),
    ],
    ids=["tree", "df_and_mapper", "df_with_nested_columns"],
)
def test_scalar_input_produces_same_flat_array_across_input_paths(
    input_data_factory,
):
    flat = main(
        main_target="input_data__flat",
        input_data=input_data_factory(),
        tt_targets=TTTargets.tree({"x": None}),
        backend="numpy",
    )
    x = np.asarray(flat[("x",)])
    np.testing.assert_array_equal(x, [5.0, 5.0, 5.0])


def test_scalar_in_input_data_flat_is_preserved_as_scalar():
    """`InputData.flat` is the advanced opt-out: it bypasses broadcasting
    so users can rely on scalar partialling for derived consumers."""
    processed = main(
        main_target="processed_data",
        input_data=InputData.flat(
            {
                ("p_id",): np.array([0, 1, 2]),
                ("x",): 5.0,
            }
        ),
        tt_targets=TTTargets.tree({"x": None}),
        backend="numpy",
    )
    assert processed["x"] == 5.0
