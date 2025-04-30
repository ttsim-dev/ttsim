from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dags.tree as dt
import optree
import pandas as pd
import yaml
from mettsim.config import RESOURCE_DIR

from ttsim import compute_taxes_and_transfers, merge_trees, set_up_policy_environment
from ttsim.config import IS_JAX_INSTALLED
from ttsim.config import numpy_or_jax as np
from ttsim.shared import to_datetime
from ttsim.ttsim_objects import GroupCreationFunction

TEST_DIR = Path(__file__).parent
# Set display options to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

if TYPE_CHECKING:
    import datetime

    from ttsim.typing import NestedDataDict, NestedInputStructureDict


class PolicyTest:
    """A class for a single policy test."""

    def __init__(
        self,
        info: NestedDataDict,
        input_tree: NestedDataDict,
        expected_output_tree: NestedDataDict,
        path: Path,
        date: datetime.date,
    ) -> None:
        self.info = info
        self.input_tree = optree.tree_map(np.array, input_tree)
        self.expected_output_tree = expected_output_tree
        self.path = path
        self.date = date

    @property
    def target_structure(self) -> NestedInputStructureDict:
        flat_target_structure = dict.fromkeys(
            dt.flatten_to_tree_paths(self.expected_output_tree)
        )
        return dt.unflatten_from_tree_paths(flat_target_structure)

    @property
    def name(self) -> str:
        return self.path.relative_to(TEST_DIR / "test_data").as_posix()


def execute_test(test: PolicyTest, jit: bool = False) -> None:
    environment = set_up_policy_environment(date=test.date, resource_dir=RESOURCE_DIR)

    if IS_JAX_INSTALLED:
        ids = dict.fromkeys(
            {f"{g}_id" for g in environment.grouping_levels}.intersection(
                {
                    g
                    for g, t in environment.raw_objects_tree.items()
                    if isinstance(t, GroupCreationFunction)
                }
            )
        )
        result_ids = compute_taxes_and_transfers(
            data_tree=test.input_tree,
            environment=environment,
            targets_tree=ids,
            jit=False,
        )
        data_tree = merge_trees(test.input_tree, result_ids)
    else:
        data_tree = test.input_tree

    result = compute_taxes_and_transfers(
        data_tree=data_tree,
        environment=environment,
        targets_tree=test.target_structure,
        jit=jit,
    )

    flat_result = dt.flatten_to_qual_names(result)
    flat_expected_output_tree = dt.flatten_to_qual_names(test.expected_output_tree)

    if flat_expected_output_tree:
        expected_df = pd.DataFrame(flat_expected_output_tree)
        result_df = pd.DataFrame(flat_result)
        if IS_JAX_INSTALLED:
            for i in [i for i in ids if i in expected_df]:
                result_df = pd.concat([result_df, pd.Series(result_ids[i])], axis=1)
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


def get_policy_test_ids_and_cases() -> dict[str, PolicyTest]:
    all_policy_tests = load_policy_test_data("")
    return {policy_test.name: policy_test for policy_test in all_policy_tests}


def load_policy_test_data(policy_name: str) -> list[PolicyTest]:
    out = []

    for path_to_yaml in (TEST_DIR / "test_data" / policy_name).glob("**/*.yaml"):
        if _is_skipped(path_to_yaml):
            continue

        with path_to_yaml.open("r", encoding="utf-8") as file:
            raw_test_data: NestedDataDict = yaml.safe_load(file)

        out.extend(
            _get_policy_tests_from_raw_test_data(
                raw_test_data=raw_test_data,
                path_to_yaml=path_to_yaml,
            )
        )

    return out


def _is_skipped(test_file: Path) -> bool:
    return "skip" in test_file.stem or "skip" in test_file.parent.name


def _get_policy_tests_from_raw_test_data(
    raw_test_data: NestedDataDict, path_to_yaml: Path
) -> list[PolicyTest]:
    """Get a list of PolicyTest objects from raw test data.

    Args:
        raw_test_data: The raw test data.
        path_to_yaml: The path to the YAML file.

    Returns:
        A list of PolicyTest objects.
    """
    test_info: NestedDataDict = raw_test_data.get("info", {})
    inputs: NestedDataDict = raw_test_data.get("inputs", {})
    input_tree: NestedDataDict = dt.unflatten_from_tree_paths(
        {
            k: pd.Series(v)
            for k, v in dt.flatten_to_tree_paths(
                merge_trees(inputs.get("provided", {}), inputs.get("assumed", {}))
            ).items()
        }
    )

    expected_output_tree: NestedDataDict = dt.unflatten_from_tree_paths(
        {
            k: pd.Series(v)
            for k, v in dt.flatten_to_tree_paths(
                raw_test_data.get("outputs", {})
            ).items()
        }
    )

    date: datetime.date = to_datetime(path_to_yaml.parent.name)

    return [
        PolicyTest(
            info=test_info,
            input_tree=input_tree,
            expected_output_tree=expected_output_tree,
            path=path_to_yaml,
            date=date,
        )
    ]
