from __future__ import annotations

import datetime
from typing import Any

import pytest
from mettsim import middle_earth

from ttsim import Labels, MainTarget, OrigPolicyObjects, TTTargets, main, plot
from ttsim.entry_point import load_flat_interface_functions_and_inputs
from ttsim.interface_dag_elements.interface_node_objects import (
    InputDependentInterfaceFunction,
)
from ttsim.plot.dag.interface import INTERFACE_COLORMAP
from ttsim.plot.dag.shared import (
    _find_color_for_qname,
    _matches_glob_pattern,
    _pattern_specificity,
)
from ttsim.plot.dag.tt import _get_tt_dag_with_node_metadata
from ttsim.tt import (
    PolicyInput,
    ScalarParam,
    param_function,
    policy_function,
)


def get_required_policy_env_objects(policy_date: datetime.date) -> dict[str, Any]:
    return {
        "policy_year": ScalarParam(
            value=policy_date.year,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_month": ScalarParam(
            value=policy_date.month,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_day": ScalarParam(
            value=policy_date.day,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "evaluation_year": PolicyInput(
            leaf_name="evaluation_year",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation year, will typically be set via `main`.",
        ),
        "evaluation_month": PolicyInput(
            leaf_name="evaluation_month",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation month, will typically be set via `main`.",
        ),
        "evaluation_day": PolicyInput(
            leaf_name="evaluation_day",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation day, will typically be set via `main`.",
        ),
    }


SOME_PARAM_OBJECT = ScalarParam(
    value=111,
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    unit=None,
    reference_period=None,
    name={"de": ""},
    description={"de": ""},
)


@param_function(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
def some_param_function():
    return 1


@policy_function(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
def some_policy_function():
    return 1


@policy_function(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
def some_policy_function_depending_on_derived_param(some_param_y: float) -> float:
    return some_param_y + 1


@pytest.mark.parametrize(
    "include_fail_and_warn_nodes",
    [
        True,
        False,
    ],
)
def test_plot_full_interface_dag(include_fail_and_warn_nodes):
    plot.dag.interface(include_fail_and_warn_nodes=include_fail_and_warn_nodes)


def test_all_namespaces_are_colored_in_interface_dag():
    flat_interface_functions_and_inputs = load_flat_interface_functions_and_inputs()

    paths_not_covered_by_colormap = set()
    for orig_path, obj in flat_interface_functions_and_inputs.items():
        if isinstance(obj, InputDependentInterfaceFunction):
            path = (*orig_path[:-1], obj.leaf_name)
        else:
            path = orig_path
        variants_expected_in_colormap = {path[: n + 1] for n in range(len(path))}
        if not any(p in INTERFACE_COLORMAP for p in variants_expected_in_colormap):
            paths_not_covered_by_colormap.add(path)

    assert not paths_not_covered_by_colormap


def test_input_dependent_interface_functions_with_same_path_have_same_docstring():
    """All input dependent interface functions with the same path must have the same
    docstring.
    """
    flat_interface_functions_and_inputs = load_flat_interface_functions_and_inputs()

    path_to_idifs: dict[tuple[str, ...], list[InputDependentInterfaceFunction]] = {}
    for orig_path, idif in flat_interface_functions_and_inputs.items():
        if isinstance(idif, InputDependentInterfaceFunction):
            path = (*orig_path[:-1], idif.leaf_name)
            if path in path_to_idifs:
                path_to_idifs[path].append(idif)
            else:
                path_to_idifs[path] = [idif]

    for path, idifs in path_to_idifs.items():
        if any(idifs[0].__doc__ != idif.__doc__ for idif in idifs):
            raise ValueError(
                "Input dependent interface functions with the same path must have the "
                f"same docstring.\n\nOffending path: {path}\n\n"
                f"Variants:\n\n" + "\n".join(idif.__name__ for idif in idifs)  # ty: ignore[unresolved-attribute]
            )


@pytest.mark.parametrize(
    (
        "selection_type",
        "selection_depth",
        "primary_nodes",
        "expected_nodes",
    ),
    [
        (
            "ancestors",
            1,
            {"payroll_tax__amount_y"},
            [
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            "ancestors",
            1,
            {"payroll_tax__amount_m"},
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            "ancestors",
            2,
            {"payroll_tax__amount_m"},
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            "ancestors",
            1,
            {"payroll_tax__amount_m", "property_tax__amount_m"},
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "property_tax__amount_m",
                "property_tax__amount_y",
            ],
        ),
        (
            "ancestors",
            None,
            {"property_tax__amount_m"},
            [
                "evaluation_year",
                "property_tax__acre_size_in_hectares",
                "property_tax__acre_size_in_hectares_after_cap",
                "evaluation_year",
                "property_tax__tax_schedule",
                "property_tax__year_from_which_cap_is_applied",
                "property_tax__cap_in_hectares",
                "property_tax__amount_y",
                "property_tax__amount_m",
            ],
        ),
        (
            "neighbors",
            1,
            {"payroll_tax__amount_y"},
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            "neighbors",
            1,
            {"payroll_tax__amount_m"},
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            "neighbors",
            2,
            {"payroll_tax__amount_m"},
            [
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            "neighbors",
            1,
            {"payroll_tax__amount_m", "property_tax__amount_m"},
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "property_tax__amount_m",
                "property_tax__amount_y",
            ],
        ),
        (
            "descendants",
            1,
            {"payroll_tax__amount_y"},
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            "descendants",
            1,
            {"payroll_tax__amount_m"},
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
            ],
        ),
        (
            "descendants",
            2,
            {"payroll_tax__amount_m"},
            [
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
            ],
        ),
        (
            "descendants",
            1,
            {"payroll_tax__amount_m", "property_tax__amount_m"},
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "property_tax__amount_m",
            ],
        ),
        (
            "descendants",
            None,
            {"housing_benefits__income__amount_m"},
            [
                "housing_benefits__amount_m_fam",
                "housing_benefits__eligibility__requirement_fulfilled_fam",
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
            ],
        ),
        (
            "all_paths",
            None,
            {"payroll_tax__income__amount_y", "payroll_tax__amount_y"},
            [
                "payroll_tax__income__amount_y",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
            ],
        ),
        (
            "all_paths",
            None,
            {"payroll_tax__income__amount_y", "property_tax__amount_y"},
            [
                "payroll_tax__income__amount_y",
                "property_tax__amount_y",
            ],
        ),
    ],
)
def test_node_selection(selection_type, selection_depth, primary_nodes, expected_nodes):
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        primary_nodes=primary_nodes,
        policy_date_str="2025-01-01",
        selection_type=selection_type,
        selection_depth=selection_depth,
        include_params=True,
    )
    assert set(dag.nodes()) == set(expected_nodes)


@pytest.mark.parametrize(
    (
        "include_params",
        "policy_environment",
        "expected_nodes",
    ),
    [
        (
            True,
            {
                "some_param": SOME_PARAM_OBJECT,
                "some_param_function": some_param_function,
                "some_policy_function": some_policy_function,
                **get_required_policy_env_objects(
                    policy_date=datetime.date(2025, 1, 1)
                ),
            },
            [
                "some_param",
                "some_param_function",
                "some_policy_function",
            ],
        ),
        (
            False,
            {
                "some_param": SOME_PARAM_OBJECT,
                "some_param_function": some_param_function,
                "some_policy_function": some_policy_function,
                **get_required_policy_env_objects(
                    policy_date=datetime.date(2025, 1, 1)
                ),
            },
            [
                "some_policy_function",
            ],
        ),
        (
            False,
            {
                "a": {
                    "some_param": SOME_PARAM_OBJECT,
                    "some_param_function": some_param_function,
                    "some_policy_function": some_policy_function,
                },
                **get_required_policy_env_objects(
                    policy_date=datetime.date(2025, 1, 1)
                ),
            },
            [
                "a__some_policy_function",
            ],
        ),
        (
            False,
            {
                "some_param_m": SOME_PARAM_OBJECT,
                "some_param_function": some_param_function,
                "some_policy_function_depending_on_derived_param": some_policy_function_depending_on_derived_param,  # noqa: E501
                **get_required_policy_env_objects(
                    policy_date=datetime.date(2025, 1, 1)
                ),
            },
            [
                "some_policy_function_depending_on_derived_param",
            ],
        ),
    ],
)
def test_params_are_removed_from_dag(
    include_params, policy_environment, expected_nodes
):
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        policy_date_str="2025-01-01",
        policy_environment=policy_environment,
        include_params=include_params,
    )
    assert set(dag.nodes()) == set(expected_nodes)


def test_orphaned_dates_are_removed_from_dag():
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        policy_date_str="2025-01-01",
        include_params=True,
    )
    assert "evaluation_day" not in dag.nodes()
    assert "policy_day" not in dag.nodes()


def test_input_data_overrides_nodes_in_plotting_dag():
    dag = main(
        main_target=MainTarget.specialized_environment_for_plotting_and_templates.complete_tt_dag,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        tt_targets=TTTargets.qname({"wealth_tax__amount_y"}),
        labels=Labels.input_columns({"wealth_tax__exempt_from_wealth_tax"}),
        include_warn_nodes=False,
    )
    assert "wealth_tax__exempt_from_wealth_tax" in dag.nodes()
    assert "wealth_tax__amount_y" in dag.nodes()
    assert "wealth_fam" not in dag.nodes()
    assert "wealth_kin" not in dag.nodes()
    assert "wealth" in dag.nodes()


def test_can_create_template_with_selection_and_input_data_from_tt():
    plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        labels=Labels.input_columns({"payroll_tax__amount_m"}),
        selection_type="ancestors",
        selection_depth=1,
        node_colormap=middle_earth.COLORMAP,
    )


def test_can_pass_plotly_kwargs_to_tt():
    plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        labels=Labels.input_columns({"payroll_tax__amount_m"}),
        selection_type="ancestors",
        selection_depth=1,
        node_colormap=middle_earth.COLORMAP,
        title="Test DAG Plot",
        width=200,
        height=800,
        showlegend=True,
        hovermode="closest",
    )


def test_fail_if_selection_type_is_all_paths_and_less_than_two_primary_nodes():
    with pytest.raises(
        ValueError, match="you must provide at least two\nprimary nodes"
    ):
        plot.dag.tt(
            root=middle_earth.ROOT_PATH,
            primary_nodes={"payroll_tax__amount_y"},
            selection_type="all_paths",
            policy_date_str="2025-01-01",
            node_colormap=middle_earth.COLORMAP,
        )


def test_fail_if_invalid_selection_type():
    with pytest.raises(
        ValueError, match="Invalid selection type: invalid_selection_type"
    ):
        plot.dag.tt(
            root=middle_earth.ROOT_PATH,
            primary_nodes={"payroll_tax__amount_y"},
            selection_type="invalid_selection_type",  # ty: ignore[invalid-argument-type]
            policy_date_str="2025-01-01",
            node_colormap=middle_earth.COLORMAP,
        )


def test_node_colormap_functionality():
    """Test that node_colormap parameter works correctly for both tt and interface
    functions.
    """

    # Test with top-level namespace coloring
    top_level_colormap = {
        ("housing_benefits",): "#ff0000",
        ("payroll_tax",): "#00ff00",
        ("wealth_tax",): "#0000ff",
    }

    # Test tt function with colormap
    fig_tt = plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        node_colormap=top_level_colormap,
    )
    assert fig_tt is not None

    # Test interface function with colormap
    fig_interface = plot.dag.interface(
        node_colormap=top_level_colormap,
    )
    assert fig_interface is not None

    # Test with hierarchical namespace coloring
    hierarchical_colormap = {
        ("housing_benefits",): "#ff0000",
        (
            "housing_benefits",
            "eligibility",
        ): "#ff8888",
        (
            "housing_benefits",
            "income",
        ): "#ff4444",
        ("payroll_tax",): "#00ff00",
        (
            "payroll_tax",
            "child_tax_credit",
        ): "#88ff88",
    }

    # Test tt function with hierarchical colormap
    fig_tt_hierarchical = plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        node_colormap=hierarchical_colormap,
    )
    assert fig_tt_hierarchical is not None

    # Test interface function with hierarchical colormap
    fig_interface_hierarchical = plot.dag.interface(
        node_colormap=hierarchical_colormap,
    )
    assert fig_interface_hierarchical is not None


def test_node_colormap_fallback_to_default():
    """Test that nodes not in the colormap fall back to default colors."""

    # Partial colormap that doesn't cover all namespaces
    partial_colormap = {
        ("housing_benefits",): "#ff0000",
    }

    # Test that this doesn't raise an error and produces a valid figure
    fig = plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        node_colormap=partial_colormap,
    )
    assert fig is not None


def test_node_colormap_glob_patterns():
    """Test that node_colormap supports glob-style pattern matching."""
    # Test _matches_glob_pattern
    # Prefix matching with *
    assert _matches_glob_pattern(("wealth_housing",), ("wealth*",))
    assert _matches_glob_pattern(("wealth_financial",), ("wealth*",))
    assert not _matches_glob_pattern(("income",), ("wealth*",))

    # Suffix matching with *
    assert _matches_glob_pattern(("amount_m",), ("*_m",))
    assert _matches_glob_pattern(("income_m",), ("*_m",))
    assert not _matches_glob_pattern(("amount_y",), ("*_m",))

    # Single character matching with ?
    assert _matches_glob_pattern(("p_id",), ("p_i?",))
    assert not _matches_glob_pattern(
        ("p_idx",), ("p_i?",)
    )  # ? matches exactly one char

    # Exact matching (no wildcards)
    assert _matches_glob_pattern(("housing_benefits",), ("housing_benefits",))
    assert not _matches_glob_pattern(("housing",), ("housing_benefits",))

    # Multi-level patterns
    assert _matches_glob_pattern(
        ("housing_benefits", "eligibility"), ("housing_benefits", "*")
    )
    assert _matches_glob_pattern(
        ("housing_benefits", "income_check"), ("housing_benefits", "*_check")
    )

    # Top-level special case
    assert _matches_glob_pattern(("p_id",), ("top-level",))
    assert not _matches_glob_pattern(("housing_benefits", "amount"), ("top-level",))

    # Test _pattern_specificity - longer patterns should score higher
    short_pattern = ("housing*",)
    long_pattern = ("housing_benefits", "*")
    assert _pattern_specificity(("housing_benefits", "amount"), long_pattern) > (
        _pattern_specificity(("housing_benefits", "amount"), short_pattern)
    )

    # Exact matches should score higher than wildcard matches
    exact_pattern = ("housing_benefits",)
    wildcard_pattern = ("housing*",)
    assert _pattern_specificity(("housing_benefits",), exact_pattern) > (
        _pattern_specificity(("housing_benefits",), wildcard_pattern)
    )

    # Test _find_color_for_qname
    colormap = {
        ("wealth*",): "blue",
        ("wealth_housing",): "navy",  # More specific, should win
        ("*_m",): "orange",
        ("housing_benefits",): "green",
        ("housing_benefits", "*"): "lightgreen",
        ("top-level",): "gray",
    }

    # Exact match takes precedence over wildcard
    assert _find_color_for_qname("wealth_housing", colormap) == "navy"

    # Wildcard prefix match
    assert _find_color_for_qname("wealth_financial", colormap) == "blue"

    # Wildcard suffix match
    assert _find_color_for_qname("income_m", colormap) == "orange"

    # Hierarchical pattern
    assert _find_color_for_qname("housing_benefits__eligibility", colormap) == (
        "lightgreen"
    )

    # Top-level fallback
    assert _find_color_for_qname("p_id", colormap) == "gray"

    # No match falls back to default
    assert _find_color_for_qname("unknown__nested", colormap) == "black"


def test_node_colormap_glob_patterns_in_plot():
    """Test that glob patterns work in actual DAG plots."""

    # Use glob patterns in colormap
    glob_colormap = {
        ("payroll*",): "#ff0000",  # Match payroll_tax and any payroll_ prefix
        ("housing_benefits", "*"): "#00ff00",  # Match all under housing_benefits
        ("*_tax",): "#0000ff",  # Match anything ending in _tax
        ("top-level",): "#888888",
    }

    # Test that this works in an actual plot
    fig = plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        selection_type="neighbors",
        node_colormap=glob_colormap,
    )
    assert fig is not None


def test_node_colormap_doublestar_patterns():
    """Test that ** patterns match any number of path segments."""
    # ** at the beginning - match suffix at any depth
    assert _matches_glob_pattern(("betrag_m_bg",), ("**", "*_bg"))
    assert _matches_glob_pattern(("bürgergeld", "betrag_m_bg"), ("**", "*_bg"))
    assert _matches_glob_pattern(
        ("bürgergeld", "einkommen", "betrag_m_bg"), ("**", "*_bg")
    )
    assert not _matches_glob_pattern(("bürgergeld", "betrag_m"), ("**", "*_bg"))

    # ** in the middle - match prefix and suffix
    assert _matches_glob_pattern(
        ("bürgergeld", "einkommen", "betrag_m"), ("bürgergeld", "**", "*_m")
    )
    assert _matches_glob_pattern(
        ("bürgergeld", "a", "b", "c", "betrag_m"), ("bürgergeld", "**", "*_m")
    )
    # ** can match zero segments
    assert _matches_glob_pattern(
        ("bürgergeld", "betrag_m"), ("bürgergeld", "**", "*_m")
    )
    assert not _matches_glob_pattern(
        ("wohngeld", "betrag_m"), ("bürgergeld", "**", "*_m")
    )

    # ** at the end - match any descendants
    assert _matches_glob_pattern(("bürgergeld",), ("bürgergeld", "**"))
    assert _matches_glob_pattern(("bürgergeld", "x"), ("bürgergeld", "**"))
    assert _matches_glob_pattern(("bürgergeld", "x", "y", "z"), ("bürgergeld", "**"))

    # Just ** matches everything
    assert _matches_glob_pattern(("anything",), ("**",))
    assert _matches_glob_pattern(("a", "b", "c"), ("**",))

    # Specificity: patterns without ** should win over patterns with **
    tp = ("bürgergeld", "betrag_m_bg")
    specific_pattern = ("bürgergeld", "*_bg")
    doublestar_pattern = ("**", "*_bg")
    assert _pattern_specificity(tp, specific_pattern) > _pattern_specificity(
        tp, doublestar_pattern
    )

    # Test _find_color_for_qname with ** patterns
    colormap = {
        ("**", "*_bg"): "purple",  # Match any _bg at any depth
        ("bürgergeld", "*"): "green",  # More specific for bürgergeld direct children
        ("bürgergeld", "betrag_m_bg"): "darkgreen",  # Most specific
        ("top-level",): "gray",
    }

    # Most specific wins
    assert _find_color_for_qname("bürgergeld__betrag_m_bg", colormap) == "darkgreen"

    # Namespace pattern wins over **
    assert _find_color_for_qname("bürgergeld__einkommen_bg", colormap) == "green"

    # ** pattern catches nested _bg
    assert _find_color_for_qname("wohngeld__nested__amount_bg", colormap) == "purple"

    # ** pattern catches top-level _bg
    assert _find_color_for_qname("some_var_bg", colormap) == "purple"


def test_node_colormap_doublestar_in_plot():
    """Test that ** patterns work in actual DAG plots."""
    doublestar_colormap = {
        ("**", "*_y"): "#ff0000",  # Match any yearly variable at any depth
        ("housing_benefits",): "#00ff00",
        ("top-level",): "#888888",
    }

    fig = plot.dag.tt(
        root=middle_earth.ROOT_PATH,
        primary_nodes={"payroll_tax__amount_y"},
        policy_date_str="2025-01-01",
        selection_type="ancestors",
        selection_depth=2,
        node_colormap=doublestar_colormap,
    )
    assert fig is not None
