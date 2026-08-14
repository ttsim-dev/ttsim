from __future__ import annotations

import datetime
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import dags
import dags.tree as dt
import optree
import pandas as pd
import yaml

from ttsim import main, merge_trees
from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.interface_dag_elements.shared import to_datetime
from ttsim.interface_dag_elements.specialized_environment_for_plotting_and_templates import (  # noqa: E501
    dummy_callable,
)
from ttsim.main_args import InputData, OrigPolicyObjects, TTTargets
from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    CompositeUnit,
    TTSIMUnit,
    UnitSystem,
    _registered_currencies,
    _unit_builder_levels,
)
from ttsim.typing import (
    FlatColumnObjectsParamFunctions,
    FlatOrigParamSpecs,
    NestedData,
    NestedInputStructureDict,
    PolicyEnvironment,
)
from ttsim.unit_validation import (
    fail_if_environment_units_are_inconsistent,
    fail_if_environment_units_are_missing,
)

# Set display options to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


@contextmanager
def isolated_unit_vocabulary() -> Iterator[None]:
    """Restore unit registrations performed inside the context on exit."""
    saved_currencies = set(_registered_currencies)
    saved_tokens = set(_ALLOWED_UNIT_TOKENS)
    saved_bases = set(vars(TTSIMUnit))
    saved_levels = set(_unit_builder_levels)
    saved_steps = set(vars(CompositeUnit))
    try:
        yield
    finally:
        _registered_currencies.clear()
        _registered_currencies.update(saved_currencies)
        _ALLOWED_UNIT_TOKENS.clear()
        _ALLOWED_UNIT_TOKENS.update(saved_tokens)
        _unit_builder_levels.clear()
        _unit_builder_levels.update(saved_levels)
        for base in set(vars(TTSIMUnit)) - saved_bases:
            delattr(TTSIMUnit, base)
        for step in set(vars(CompositeUnit)) - saved_steps:
            delattr(CompositeUnit, step)


def get_policy_date_partition(
    orig_policy_objects: dict[
        str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
    ],
    unit_system: UnitSystem | None = None,
) -> list[datetime.date]:
    """The dates a policy package must be validated at — one per structural regime.

    The resolved environment changes not only where a function starts or stops, but
    also where a parameter entry or the statutory currency does: a parameter-only
    change opens a regime of its own even though every function stays active, and a
    partition built from function dates alone never assembles it (GEP 10). Each
    returned date is the left endpoint of one regime:

    - every column object's and param function's start date, and the day after each
      inclusive end date;
    - every dated parameter entry — where a value, unit, currency, or leaf set may
      change;
    - every statutory-currency start date of ``unit_system``, where one is given.

    Boundaries outside the package's own date domain — the span from its earliest
    start date to the day after its latest end date — are dropped: no environment
    exists there to validate. Rounding specifications carry the dates of the
    function they sit on and so contribute no boundary of their own.

    Args:
        orig_policy_objects: The package's ``column_objects_and_param_functions``
            and ``param_specs``, as returned by `main`.
        unit_system: The package's unit system, whose statutory-currency
            transitions are boundaries too.

    Returns:
        The regime start dates, sorted and deduplicated.

    """
    column_objects = orig_policy_objects["column_objects_and_param_functions"]
    start_dates = {
        obj.start_date  # ty: ignore[unresolved-attribute]
        for obj in column_objects.values()
    }
    if not start_dates:
        return []
    end_dates = {
        obj.end_date + datetime.timedelta(days=1)  # ty: ignore[unresolved-attribute]
        for obj in column_objects.values()
    }
    param_dates = {
        key
        for spec in orig_policy_objects.get("param_specs", {}).values()
        for key in spec
        if isinstance(key, datetime.date)
    }
    currency_dates = (
        {date for date, _ in unit_system.statutory_currency_by_start_date}
        if unit_system is not None
        else set()
    )
    domain_start = min(start_dates)
    domain_end = max(end_dates)
    return sorted(
        date
        for date in start_dates | end_dates | param_dates | currency_dates
        if domain_start <= date <= domain_end
    )


@lru_cache(maxsize=100)
def cached_policy_environment(
    policy_date: datetime.date,
    root: Path,
    backend: Literal["numpy", "jax"],
    unit_system: UnitSystem,
) -> PolicyEnvironment:
    return main(
        main_target="policy_environment",
        policy_date=policy_date,
        orig_policy_objects=OrigPolicyObjects.root(root),
        unit_system=unit_system,
        backend=backend,
        include_fail_nodes=True,
        include_warn_nodes=False,
    )


class PolicyTest:
    """A class for a single policy test."""

    __slots__ = (
        "expected_output_tree",
        "info",
        "input_tree",
        "path",
        "policy_cases_root",
        "policy_date",
        "xnp",
    )

    def __init__(
        self,
        info: dict[str, Any],
        input_tree: NestedData,
        expected_output_tree: NestedData,
        path: Path,
        policy_date: datetime.date,
        policy_cases_root: Path,
        xnp: ModuleType,
    ) -> None:
        self.info = info
        self.input_tree = optree.tree_map(xnp.array, input_tree)  # ty: ignore[invalid-argument-type]
        self.expected_output_tree = expected_output_tree
        self.path = path
        self.policy_date = policy_date
        self.policy_cases_root = policy_cases_root
        self.xnp = xnp

    @property
    def target_structure(self) -> NestedInputStructureDict:
        flat_target_structure = dict.fromkeys(
            dt.flatten_to_tree_paths(self.expected_output_tree),
        )
        return dt.unflatten_from_tree_paths(flat_target_structure)

    @property
    def name(self) -> str:
        return self.path.relative_to(self.policy_cases_root).as_posix()


def execute_test(
    test: PolicyTest,
    root: Path,
    backend: Literal["numpy", "jax"],
    unit_system: UnitSystem,
    default_data_currency: str | None = None,
) -> None:
    environment = cached_policy_environment(
        policy_date=test.policy_date,
        root=root,
        backend=backend,
        unit_system=unit_system,
    )
    if test.target_structure:
        result_df = main(
            main_target="results__df_with_nested_columns",
            input_data=InputData.tree(test.input_tree),  # ty: ignore[invalid-argument-type]
            policy_environment=environment,
            policy_date=test.policy_date,
            tt_targets=TTTargets.tree(test.target_structure),
            rounding=True,
            backend=backend,
            unit_system=unit_system,
            data_currency=test.info.get("data_currency", default_data_currency),
            include_fail_nodes=True,
            include_warn_nodes=False,
        )

        if test.expected_output_tree:
            expected_df = nested_data_to_df_with_nested_columns(
                nested_data_to_convert=test.expected_output_tree,
                index=pd.Index(test.input_tree["p_id"], name="p_id"),
            )
            try:
                pd.testing.assert_frame_equal(
                    result_df.sort_index(axis="columns"),
                    expected_df.sort_index(axis="columns"),
                    atol=test.info["precision_atol"],
                    check_dtype=False,
                )
            except AssertionError as e:
                assert set(result_df.columns) == set(expected_df.columns)  # noqa: S101
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
""",
                ) from e


def load_policy_cases(
    policy_cases_root: Path,
    policy_name: str,
    xnp: ModuleType,
) -> dict[str, PolicyTest]:
    """Load all tests found by recursively searching

        policy_cases_root / policy_name

    for yaml files.

    If `policy_name` is empty, all tests found in `policy_cases_root` are loaded.
    """
    out = {}
    for path_to_yaml in (policy_cases_root / policy_name).glob("**/*.yaml"):
        if _is_skipped(path_to_yaml):
            continue

        with path_to_yaml.open("r", encoding="utf-8") as file:
            raw_test_data: dict[str, Any] = yaml.safe_load(file)

            this_test = _get_policy_test_from_raw_test_data(
                policy_cases_root=policy_cases_root,
                raw_test_data=raw_test_data,
                path_to_yaml=path_to_yaml,
                xnp=xnp,
            )
            out[this_test.name] = this_test

    return out


def _is_skipped(test_file: Path) -> bool:
    return "skip" in test_file.stem or "skip" in test_file.parent.name


def _get_policy_test_from_raw_test_data(
    policy_cases_root: Path,
    path_to_yaml: Path,
    raw_test_data: dict[str, Any],
    xnp: ModuleType,
) -> PolicyTest:
    """Get a list of PolicyTest objects from raw test data.

    Args:
        raw_test_data: The raw test data.
        path_to_yaml: The path to the YAML file.

    Returns:
        A list of PolicyTest objects.
    """
    test_info: dict[str, Any] = raw_test_data.get("info", {})
    input_tree: NestedData = dt.unflatten_from_tree_paths(
        {
            k: xnp.array(v)
            for k, v in dt.flatten_to_tree_paths(
                merge_trees(
                    left=raw_test_data["inputs"].get("provided", {}),
                    right=raw_test_data["inputs"].get("assumed", {}),
                ),
            ).items()
        },
    )

    expected_output_tree: NestedData = dt.unflatten_from_tree_paths(
        {
            k: xnp.array(v)
            for k, v in dt.flatten_to_tree_paths(
                raw_test_data.get("outputs", {}),
            ).items()
        },
    )

    policy_date: datetime.date = to_datetime(path_to_yaml.parent.name)

    return PolicyTest(
        info=test_info,
        input_tree=input_tree,
        expected_output_tree=expected_output_tree,
        path=path_to_yaml,
        policy_date=policy_date,
        policy_cases_root=policy_cases_root,
        xnp=xnp,
    )


def check_env_completeness(
    name: str,
    policy_date: datetime.date,
    orig_policy_objects: dict[
        str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
    ],
    unit_system: UnitSystem,
) -> None:
    qname_env_with_derived_functions = main(
        main_target="specialized_environment_for_plotting_and_templates__without_tree_logic_and_with_derived_functions",
        policy_date=policy_date,
        orig_policy_objects=OrigPolicyObjects(**orig_policy_objects),
        unit_system=unit_system,
        backend="numpy",
    )
    all_nodes = {
        qn: dummy_callable(obj=n, leaf_name=dt.tree_path_from_qname(qn)[-1])
        if not callable(n)
        else n
        for qn, n in qname_env_with_derived_functions.items()
    }
    f = dags.concatenate_functions(
        functions=all_nodes,
        targets=list(qname_env_with_derived_functions.keys()),
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = set(inspect.signature(f).parameters) - {
        "backend",
        "xnp",
        "dnp",
        "num_segments",
        "evaluation_year",
        "evaluation_month",
        "evaluation_day",
    }
    if args:
        raise ValueError(
            f"{name}'s full DAG should include all root nodes but the following inputs "
            "are missing in the specialized policy environment:"
            f"\n\n{format_list_linewise(args)}\n\n"
            "Please add corresponding elements. Typically, these will be "
            "`@policy_input()`s or parameters in the yaml files."
        )


def check_policy_environment_units(
    policy_date: datetime.date,
    orig_policy_objects: dict[
        str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
    ],
    unit_system: UnitSystem,
) -> None:
    """Run the unit checks over the full environment at a policy date.

    Builds the data-independent specialized environment (all derivable nodes
    included) and runs both environment-level unit checks: mandatory units and
    the conservative body/edge verification. Intended to be parametrized over
    all policy dates of a country package as a CI test.

    Raises:
        UnitDefinitionError: If any active node lacks a mandatory unit.
        UnitConsistencyError: If any function body infers a concrete unit
            that disagrees with its declaration.
    """
    targets = main(
        main_targets=[
            "specialized_environment_for_plotting_and_templates__without_tree_logic_and_with_derived_functions",
            "labels__grouping_levels",
        ],
        policy_date=policy_date,
        orig_policy_objects=OrigPolicyObjects(**orig_policy_objects),
        unit_system=unit_system,
        backend="numpy",
    )
    env = targets["specialized_environment_for_plotting_and_templates"][
        "without_tree_logic_and_with_derived_functions"
    ]
    grouping_levels = targets["labels"]["grouping_levels"]
    fail_if_environment_units_are_missing(env)
    fail_if_environment_units_are_inconsistent(
        env=env, grouping_levels=grouping_levels, unit_system=unit_system
    )
