from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import dags.tree as dt
import pandas as pd
import yaml

from ttsim import merge_trees

TEST_DIR = Path(__file__).parent / "test_data"

if TYPE_CHECKING:
    from ttsim import NestedDataDict, NestedInputStructureDict


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
        self.input_tree = input_tree
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
    def test_name(self) -> str:
        return self.path.relative_to(TEST_DIR / "test_data").as_posix()


def execute_test(test: PolicyTest):
    from pandas.testing import assert_frame_equal

    from _gettsim_tests._helpers import cached_set_up_policy_environment
    from ttsim import compute_taxes_and_transfers

    environment = cached_set_up_policy_environment(date=test.date)

    result = compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree=test.target_structure,
        foreign_keys=FOREIGN_KEYS,
        supported_groupings=SUPPORTED_GROUPINGS,
    )

    flat_result = dt.flatten_to_qual_names(result)
    flat_expected_output_tree = dt.flatten_to_qual_names(test.expected_output_tree)

    if flat_expected_output_tree:
        result_dataframe = pd.DataFrame(flat_result)
        expected_dataframe = pd.DataFrame(flat_expected_output_tree)
        assert_frame_equal(
            result_dataframe,
            expected_dataframe,
            atol=test.info["precision"],
            check_dtype=False,
        )


def get_policy_test_ids_and_cases() -> dict[str, PolicyTest]:
    all_policy_tests = load_policy_test_data("")
    return {policy_test.test_name: policy_test for policy_test in all_policy_tests}


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


def get_test_data_as_tree(test_data: NestedDataDict) -> NestedDataDict:
    provided_inputs = test_data["inputs"].get("provided", {})
    assumed_inputs = test_data["inputs"].get("assumed", {})

    unflattened_dict = {}
    unflattened_dict["inputs"] = {}
    unflattened_dict["outputs"] = {}

    if provided_inputs:
        unflattened_dict["inputs"]["provided"] = dt.unflatten_from_qual_names(
            provided_inputs
        )
    else:
        unflattened_dict["inputs"]["provided"] = {}
    if assumed_inputs:
        unflattened_dict["inputs"]["assumed"] = dt.unflatten_from_qual_names(
            assumed_inputs
        )
    else:
        unflattened_dict["inputs"]["assumed"] = {}

    unflattened_dict["outputs"] = dt.unflatten_from_qual_names(test_data["outputs"])

    return unflattened_dict["inputs"], unflattened_dict["outputs"]


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

    date: datetime.date = _parse_date(path_to_yaml.parent.name)

    out = []
    if expected_output_tree == {}:
        out.append(
            PolicyTest(
                info=test_info,
                input_tree=input_tree,
                expected_output_tree={},
                path=path_to_yaml,
                date=date,
            )
        )
    else:
        for target_name, output_data in dt.flatten_to_tree_paths(
            expected_output_tree
        ).items():
            one_expected_output: NestedDataDict = dt.unflatten_from_tree_paths(
                {target_name: output_data}
            )
            out.append(
                PolicyTest(
                    info=test_info,
                    input_tree=input_tree,
                    expected_output_tree=one_expected_output,
                    path=path_to_yaml,
                    date=date,
                )
            )

    return out


def _parse_date(date: str) -> datetime.date:
    parts = date.split("-")

    if len(parts) == 1:
        return datetime.date(int(parts[0]), 1, 1)
    if len(parts) == 2:
        return datetime.date(int(parts[0]), int(parts[1]), 1)
    if len(parts) == 3:
        return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
