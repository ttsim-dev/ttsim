from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import dags.tree as dt
import optree
import pandas as pd
import yaml

from ttsim import main, merge_trees
from ttsim.config import numpy_or_jax as np
from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.shared import to_datetime

# Set display options to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

if TYPE_CHECKING:
    import datetime
    from pathlib import Path

    from ttsim.interface_dag_elements.typing import (
        NestedData,
        NestedInputStructureDict,
        NestedPolicyEnvironment,
    )


@lru_cache(maxsize=100)
def cached_policy_environment(
    date: datetime.date, root: Path
) -> NestedPolicyEnvironment:
    return main(
        inputs={
            "date": date,
            "orig_policy_objects__root": root,
        },
        targets=["policy_environment"],
    )["policy_environment"]


class PolicyTest:
    """A class for a single policy test."""

    def __init__(
        self,
        info: NestedData,
        input_tree: NestedData,
        expected_output_tree: NestedData,
        path: Path,
        date: datetime.date,
        test_dir: Path,
    ) -> None:
        self.info = info
        self.input_tree = optree.tree_map(np.array, input_tree)
        self.expected_output_tree = expected_output_tree
        self.path = path
        self.date = date
        self.test_dir = test_dir

    @property
    def target_structure(self) -> NestedInputStructureDict:
        flat_target_structure = dict.fromkeys(
            dt.flatten_to_tree_paths(self.expected_output_tree)
        )
        return dt.unflatten_from_tree_paths(flat_target_structure)

    @property
    def name(self) -> str:
        return self.path.relative_to(self.test_dir / "test_data").as_posix()


def execute_test(test: PolicyTest, root: Path, jit: bool = False) -> None:  # noqa: ARG001
    environment = cached_policy_environment(date=test.date, root=root)

    if test.target_structure:
        nested_result = main(
            inputs={
                "input_data__tree": test.input_tree,
                "policy_environment": environment,
                "targets__tree": test.target_structure,
                "rounding": True,
                # "jit": jit,
            },
            targets=["results__tree"],
        )["results__tree"]
    else:
        nested_result = {}

    if test.expected_output_tree:
        expected_df = nested_data_to_df_with_nested_columns(
            nested_data_to_convert=test.expected_output_tree,
            data_with_p_id=test.input_tree,
        )
        result_df = nested_data_to_df_with_nested_columns(
            nested_data_to_convert=nested_result, data_with_p_id=test.input_tree
        )
        try:
            pd.testing.assert_frame_equal(
                result_df.sort_index(axis="columns"),
                expected_df.sort_index(axis="columns"),
                atol=test.info["precision_atol"],
                check_dtype=False,
            )
        except AssertionError as e:
            assert set(result_df.columns) == set(expected_df.columns)
            cols_with_differences = []
            for col in expected_df.columns:
                try:
                    pd.testing.assert_series_equal(
                        result_df[col],
                        expected_df[col],
                        atol=test.info["precision_atol"],
                        check_dtype=False,
                    )
                except AssertionError:
                    cols_with_differences.append(col)
            raise AssertionError(
                f"""actual != expected in columns: {cols_with_differences}.

actual[cols_with_differences]:

{result_df[cols_with_differences]}

expected[cols_with_differences]:

{expected_df[cols_with_differences]}
"""
            ) from e


def load_policy_test_data(test_dir: Path, policy_name: str) -> dict[str, PolicyTest]:
    """Load all tests found by recursively searching

        test_dir / "test_data" / policy_name

    for yaml files.

    If policy_name is empty, all tests found in test_dir / "test_data" are loaded.
    """

    out = {}
    for path_to_yaml in (test_dir / "test_data" / policy_name).glob("**/*.yaml"):
        if _is_skipped(path_to_yaml):
            continue

        with path_to_yaml.open("r", encoding="utf-8") as file:
            raw_test_data: NestedData = yaml.safe_load(file)

            this_test = _get_policy_test_from_raw_test_data(
                test_dir=test_dir,
                raw_test_data=raw_test_data,
                path_to_yaml=path_to_yaml,
            )
            out[this_test.name] = this_test

    return out


def _is_skipped(test_file: Path) -> bool:
    return "skip" in test_file.stem or "skip" in test_file.parent.name


def _get_policy_test_from_raw_test_data(
    test_dir: Path,
    path_to_yaml: Path,
    raw_test_data: NestedData,
) -> PolicyTest:
    """Get a list of PolicyTest objects from raw test data.

    Args:
        raw_test_data: The raw test data.
        path_to_yaml: The path to the YAML file.

    Returns:
        A list of PolicyTest objects.
    """
    test_info: NestedData = raw_test_data.get("info", {})
    input_tree: NestedData = dt.unflatten_from_tree_paths(
        {
            k: np.array(v)
            for k, v in dt.flatten_to_tree_paths(
                merge_trees(
                    left=raw_test_data["inputs"].get("provided", {}),
                    right=raw_test_data["inputs"].get("assumed", {}),
                )
            ).items()
        }
    )
    expected_output_tree: NestedData = dt.unflatten_from_tree_paths(
        {
            k: np.array(v)
            for k, v in dt.flatten_to_tree_paths(
                raw_test_data.get("outputs", {})
            ).items()
        }
    )

    date: datetime.date = to_datetime(path_to_yaml.parent.name)

    return PolicyTest(
        info=test_info,
        input_tree=input_tree,
        expected_output_tree=expected_output_tree,
        path=path_to_yaml,
        date=date,
        test_dir=test_dir,
    )
