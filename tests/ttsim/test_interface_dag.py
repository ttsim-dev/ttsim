from __future__ import annotations

import inspect

import dags
import pytest

from ttsim.interface_dag import (
    _fail_if_qnames_are_not_among_nodes,
    load_interface_functions_and_inputs,
)
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.interface_dag_elements.interface_node_objects import fail_or_warn_function
from ttsim.plot_dag import dummy_callable


@fail_or_warn_function(
    include_if_all_elements_present=["a"],
    include_if_any_element_present=["b"],
)
def some_fail_or_warn_function() -> None:
    pass


def test_load_interface_functions_and_inputs() -> None:
    load_interface_functions_and_inputs()


def test_interface_dag_is_complete() -> None:
    nodes = {
        p: dummy_callable(n) if not callable(n) else n
        for p, n in load_interface_functions_and_inputs().items()
    }

    f = dags.concatenate_functions(
        functions=nodes,
        targets=None,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "The full interface DAG should include all root nodes but requires inputs:"
            f"\n\n{format_list_linewise(args.keys())}"
        )


@pytest.mark.parametrize(
    ("output_qnames", "nodes", "error_match"),
    [
        (
            ["a"],
            load_interface_functions_and_inputs(),
            r'not among the interface functions or inputs:  \[\n"a", \]',
        ),
        (
            ["input_data"],
            load_interface_functions_and_inputs(),
            r'not among the interface functions or inputs:  \[\n"input_data", \]',
        ),
        (
            [],
            {
                **load_interface_functions_and_inputs(),
                "some_fail_or_warn_function": some_fail_or_warn_function,
            },
            r'not among the interface functions or inputs:  \[\n"a",     "b", \]',
        ),
    ],
)
def test_fail_if_qnames_are_not_among_nodes(output_qnames, nodes, error_match) -> None:
    with pytest.raises(ValueError, match=error_match):
        _fail_if_qnames_are_not_among_nodes(
            output_qnames=output_qnames,
            nodes=nodes,
        )
