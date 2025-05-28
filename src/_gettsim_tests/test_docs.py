from __future__ import annotations

import datetime
import inspect

import pytest

from _gettsim.config import GETTSIM_ROOT
from ttsim import PolicyInput
from ttsim.loader import orig_tree_with_column_objects_and_param_functions
from ttsim.policy_environment import active_tree_with_column_objects_and_param_functions
from ttsim.shared import remove_group_suffix


def _nice_output_list_of_strings(list_of_strings):
    my_str = "\n".join(sorted(list_of_strings))
    return f"\n\n{my_str}\n\n"


@pytest.fixture(scope="module")
def default_input_variables():
    return sorted(f for f in todo_functions_tree if isinstance(f, PolicyInput))


@pytest.fixture(scope="module")
def all_function_names():
    functions = _load_internal_functions()
    return sorted([func.leaf_name for func in functions])


@pytest.fixture(scope="module")
def time_indep_function_names(all_function_names):
    time_dependent_functions = {}
    _orig_tree_with_column_objects_and_param_functions = (
        orig_tree_with_column_objects_and_param_functions(root=GETTSIM_ROOT)
    )
    for year in range(1990, 2023):
        year_functions = active_tree_with_column_objects_and_param_functions(
            orig_tree_with_column_objects_and_param_functions=_orig_tree_with_column_objects_and_param_functions,
            date=datetime.date(year=year, month=1, day=1),
        )
        new_dict = {func.function.__name__: func.leaf_name for func in year_functions}
        time_dependent_functions = {**time_dependent_functions, **new_dict}

    # Only use time dependent function names
    time_indep_function_names = [
        (time_dependent_functions.get(c, c)) for c in sorted(all_function_names)
    ]

    # Remove duplicates
    time_indep_function_names = list(dict.fromkeys(time_indep_function_names))
    return time_indep_function_names


@pytest.mark.xfail(
    reason="Uses an internal function to load functions,"
    " which does not include derived functions."
)
def test_all_input_vars_documented(
    default_input_variables,
    time_indep_function_names,
    all_function_names,
):
    """Test if arguments of all non-internal functions are either the name of another
    function, a documented input variable, or a parameter dictionary."""
    functions = _load_internal_functions()

    # Collect arguments of all non-internal functions (do not start with underscore)
    arguments = [
        i
        for f in functions
        for i in list(inspect.signature(f).parameters)
        if not f.leaf_name.startswith("_")
    ]

    # Remove duplicates
    arguments = list(dict.fromkeys(arguments))
    defined_functions = (
        time_indep_function_names + all_function_names + default_input_variables
    )
    check = [
        c
        for c in arguments
        if (c not in defined_functions)
        and (
            remove_group_suffix(c, groupings=environment.grouping_levels)
            not in defined_functions
        )
        and (not c.endswith("_params"))
    ]

    assert not check, _nice_output_list_of_strings(check)


@pytest.mark.xfail(reason="Not able to load functions regardless of date any more.")
def test_funcs_in_doc_module_and_func_from_internal_files_are_the_same():
    documented_functions = {
        f.leaf_name
        for f in _load_functions(
            GETTSIM_ROOT / "functions" / "all_functions_for_docs.py",
            package_root=GETTSIM_ROOT,
            include_imported_functions=True,
        )
    }

    internal_function_files = [GETTSIM_ROOT.joinpath(p) for p in GETTSIM_ROOT]

    internal_functions = {
        f.leaf_name
        for f in _load_functions(
            internal_function_files,
            package_root=GETTSIM_ROOT,
            include_imported_functions=True,
        )
        if not f.original_function_name.startswith("_")
    }

    assert documented_functions == internal_functions


@pytest.mark.xfail(reason="Not able to load functions regardless of date any more.")
def test_type_hints():  # noqa: PLR0912
    """Check if output and input types of all functions coincide."""
    types = {}

    for func in _load_internal_functions():
        if func.vectorization_strategy == "not_required":
            continue

        name = func.leaf_name

        for var, internal_type in func.__annotations__.items():
            if var == "return":
                if name in types:
                    if types[name] != internal_type:
                        raise ValueError(
                            f"The return type hint of {func.original_function_name}, "
                            f"does not coincide  with the input type hint of "
                            f"another function."
                        )
                else:
                    types[name] = internal_type
            else:
                if var in TYPES_INPUT_VARIABLES:
                    if internal_type != TYPES_INPUT_VARIABLES[var]:
                        raise ValueError(
                            f"The input type hint of {var} in function "
                            f"{func.original_function_name} does not coincide with the "
                            f"standard data types provided in the config file."
                        )
                elif var in types:
                    if types[var] != internal_type:
                        raise ValueError(
                            f"The type hint of {var} in {func.original_function_name} "
                            f"does not coincide with the input type hint "
                            f"of another function."
                        )
                else:
                    types[var] = internal_type
