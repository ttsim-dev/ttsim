from __future__ import annotations

from pathlib import Path

import dags.tree as dt
import numpy

from ttsim import main
from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.testing_utils import (
    load_policy_test_data,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    policy_function,
    policy_input,
)
from ttsim.tt_dag_elements.param_objects import DictParam, ScalarParam

GETTSIM_ROOT = Path(__file__).parent.parent / "_gettsim"

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


p1 = ScalarParam(
    value=1,
    leaf_name="some_int_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_int_param",
    description="Some int param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)

p2 = DictParam(
    value={"a": 1, "b": 2},
    leaf_name="some_dict_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_dict_param",
    description="Some dict param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


@policy_input()
def inp1() -> int:
    pass


@policy_input()
def inp2() -> float:
    pass


@policy_function()
def x(inp1: int, p1: int, p2: dict[str, int]) -> int:
    return inp1 + p1 + p2["a"] + p2["b"]


@policy_function()
def y(inp2: float, p2: dict[str, int]) -> float:
    return inp2 + p2["b"]


@policy_function()
def z(a__x: int, a__y: float) -> float:
    return a__x + a__y


def test_template_all_outputs_no_inputs(backend):
    res = main(
        inputs={
            "orig_policy_objects__root": GETTSIM_ROOT,
            "rounding": True,
            "date_str": "2025-01-01",
            "backend": backend,
        },
        output_names=["labels__grouping_levels", "templates__input_data_dtypes"],
    )
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        time_units=list(TIME_UNIT_LABELS),
        grouping_levels=res["labels__grouping_levels"],
    )
    bn_to_variations = {}
    for qname in dt.qnames(res["templates__input_data_dtypes"]):
        match = pattern_all.fullmatch(qname)
        # We must not find multiple time units for the same base name and group.
        base_name = match.group("base_name")
        if base_name not in bn_to_variations:
            bn_to_variations[base_name] = [qname]
        else:
            bn_to_variations[base_name].append(qname)
    dups = {bn: v for bn, v in bn_to_variations.items() if len(v) > 1}
    if dups:
        formatted = ""
        for base_name, variations in dups.items():
            formatted += f"\n{base_name}:\n    "
            formatted += "\n    ".join(variations)
            formatted += "\n"
        raise AssertionError(f"More than one variation for base names:\n{formatted}")
