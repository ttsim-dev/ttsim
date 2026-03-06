"""Some tests for the policy_environment module."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import optree
import pandas as pd
import pytest

from ttsim import OrigPolicyObjects, main
from ttsim.interface_dag_elements.orig_policy_objects import (
    column_objects_and_param_functions,
)
from ttsim.interface_dag_elements.policy_environment import (
    _active_column_objects_and_param_functions,
    _active_param_objects,
)
from ttsim.tt import ScalarParam, policy_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        NestedColumnObjectsParamFunctions,
    )

from mettsim import middle_earth


@pytest.fixture(scope="module")
def some_int_param():
    return ScalarParam(
        value=1,
        start_date=datetime.date(2025, 1, 1),
        end_date=datetime.date(2025, 12, 31),
        name={"de": "Some int param", "en": "Some int param"},
        description={"de": "Some int param", "en": "Some int param"},
        unit=None,
        reference_period=None,
        note=None,
        reference=None,
    )


@pytest.fixture
def piecewise_spec_base():
    return {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "piecewise_linear",
        datetime.date(2020, 1, 1): {
            "intervals": [
                {"interval": "[0, 100)", "slope": 0.5, "intercept": 0},
                {"interval": "[100, 200)", "slope": 0.4, "intercept": 5},
            ],
        },
    }


def test_add_jahresanfang(xnp: ModuleType):
    spec = {
        "name": {"de": "Test", "en": "Check"},
        "description": {"de": "Nichts zu sehen", "en": "Nothing to do"},
        "type": "scalar",
        "add_jahresanfang": True,
        datetime.date(2020, 1, 1): {"value": 1},
        datetime.date(2020, 7, 1): {"value": 2},
    }
    _active_ttsim_tree_with_params = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=pd.to_datetime("2020-07-01").date(),
        xnp=xnp,
    )
    assert _active_ttsim_tree_with_params["foo"].value == 2
    assert _active_ttsim_tree_with_params["foo_jahresanfang"].value == 1


def test_input_is_recognized_as_potential_group_id(backend):
    assert "kin" in main(
        main_target="labels__grouping_levels",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=datetime.date(2020, 1, 1),
        backend=backend,
    )


def test_p_id_not_recognized_as_potential_group_id(backend):
    assert "p" not in main(
        main_target="labels__grouping_levels",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=datetime.date(2020, 1, 1),
        backend=backend,
    )


@pytest.mark.parametrize(
    ("date_string", "expected"),
    [
        ("2023-01-20", datetime.date(2023, 1, 20)),
    ],
)
def test_start_date_valid(date_string: str, expected: datetime.date):
    @policy_function(start_date=date_string)
    def test_func():
        pass

    assert test_func.start_date == expected


@pytest.mark.parametrize(
    "date_string",
    [
        "20230120",
        "20.1.2023",
        "20th January 2023",
    ],
)
def test_start_date_invalid(date_string: str):
    with pytest.raises(
        ValueError,
        match=r"neither matches the format YYYY-MM-DD nor is a datetime.date",
    ):

        @policy_function(start_date=date_string)
        def test_func():
            pass


def test_start_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.start_date == datetime.date(1900, 1, 1)


@pytest.mark.parametrize(
    ("date_string", "expected"),
    [
        ("2023-01-20", datetime.date(2023, 1, 20)),
    ],
)
def test_end_date_valid(date_string: str, expected: datetime.date):
    @policy_function(end_date=date_string)
    def test_func():
        pass

    assert test_func.end_date == expected


@pytest.mark.parametrize(
    "date_string",
    [
        "20230120",
        "20.1.2023",
        "20th January 2023",
    ],
)
def test_end_date_invalid(date_string: str):
    with pytest.raises(
        ValueError,
        match=r"neither matches the format YYYY-MM-DD nor is a datetime.date",
    ):

        @policy_function(end_date=date_string)
        def test_func():
            pass


def test_end_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.end_date == datetime.date(2099, 12, 31)


def test_active_period_is_empty():
    with pytest.raises(ValueError, match="must be before the end date"):

        @policy_function(start_date="2023-01-20", end_date="2023-01-19")
        def test_func():
            pass


@pytest.mark.parametrize(
    ("tree", "last_day", "function_name_last_day", "function_name_next_day"),
    [
        (
            {"housing_benefits": {"eligibility": {"requirement_fulfilled_fam": None}}},
            datetime.date(2019, 12, 31),
            "requirement_fulfilled_fam_not_considering_children",
            "requirement_fulfilled_fam_considering_children",
        ),
    ],
)
def test_active_tree_with_column_objects_and_param_functions(
    tree: NestedColumnObjectsParamFunctions,
    last_day: datetime.date,
    function_name_last_day: str,
    function_name_next_day: str,
):
    orig = column_objects_and_param_functions(root=middle_earth.ROOT_PATH)
    functions_last_day = _active_column_objects_and_param_functions(
        orig=orig,
        policy_date=last_day,
    )
    functions_next_day = _active_column_objects_and_param_functions(
        orig=orig,
        policy_date=last_day + datetime.timedelta(days=1),
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]  # ty: ignore[invalid-argument-type]

    assert accessor(functions_last_day).__name__ == function_name_last_day
    assert accessor(functions_next_day).__name__ == function_name_next_day


def test_piecewise_updates_previous(piecewise_spec_base, xnp: ModuleType):
    """Piecewise param with updates_previous merges intervals."""
    spec = piecewise_spec_base
    spec[datetime.date(2021, 1, 1)] = {
        "updates_previous": True,
        "intervals": [
            {"interval": "[0, 100)", "slope": 0.9, "intercept": 0},
        ],
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},
        policy_date=datetime.date(2021, 6, 1),
        xnp=xnp,
    )
    params = result["foo"].value
    # The first interval's slope should be updated to 0.9
    assert params.coefficients[0][0] == pytest.approx(0.9)
    # The second interval should remain from base
    assert params.coefficients[1][0] == pytest.approx(0.4)


def test_piecewise_no_updates_previous(piecewise_spec_base, xnp: ModuleType):
    """Piecewise param without updates_previous uses intervals directly."""
    spec = piecewise_spec_base
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},
        policy_date=datetime.date(2020, 6, 1),
        xnp=xnp,
    )
    params = result["foo"].value
    assert params.coefficients[0][0] == pytest.approx(0.5)
    assert params.coefficients[1][0] == pytest.approx(0.4)


def test_dict_updates_previous(xnp: ModuleType):
    """Dict param with updates_previous merges dicts."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "a": 1,
            "b": 2,
        },
        datetime.date(2021, 1, 1): {
            "updates_previous": True,
            "a": 10,
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2021, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"a": 10, "b": 2}


def test_dict_updates_previous_adds_new_key(xnp: ModuleType):
    """Dict updates_previous can add keys not present in the base."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "a": 1,
        },
        datetime.date(2021, 1, 1): {
            "updates_previous": True,
            "b": 2,
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2021, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"a": 1, "b": 2}


def test_dict_updates_previous_chained(xnp: ModuleType):
    """Three dates with chained updates_previous merges all."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "a": 1,
            "b": 2,
            "c": 3,
        },
        datetime.date(2021, 1, 1): {
            "updates_previous": True,
            "a": 10,
        },
        datetime.date(2022, 1, 1): {
            "updates_previous": True,
            "b": 20,
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2022, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"a": 10, "b": 20, "c": 3}


def test_dict_updates_previous_nested(xnp: ModuleType):
    """Nested dict is merged recursively via upsert_tree."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "outer": {"x": 1, "y": 2},
        },
        datetime.date(2021, 1, 1): {
            "updates_previous": True,
            "outer": {"x": 10},
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2021, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"outer": {"x": 10, "y": 2}}


def test_dict_updates_previous_queries_base_date(xnp: ModuleType):
    """Querying the base date ignores updates_previous on later dates."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "a": 1,
            "b": 2,
        },
        datetime.date(2021, 1, 1): {
            "updates_previous": True,
            "a": 10,
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2020, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"a": 1, "b": 2}


def test_dict_no_updates_previous(xnp: ModuleType):
    """Dict param without updates_previous uses current spec directly."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "a": 1,
            "b": 2,
        },
        datetime.date(2021, 1, 1): {
            "c": 3,
        },
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=datetime.date(2021, 6, 1),
        xnp=xnp,
    )
    assert result["foo"].value == {"c": 3}


def test_piecewise_updates_previous_chained(piecewise_spec_base, xnp: ModuleType):
    """Chained piecewise updates_previous across 3+ dates merges correctly."""
    spec = piecewise_spec_base
    spec[datetime.date(2021, 1, 1)] = {
        "updates_previous": True,
        "intervals": [
            {"interval": "[0, 100)", "slope": 0.9, "intercept": 0},
        ],
    }
    spec[datetime.date(2022, 1, 1)] = {
        "updates_previous": True,
        "intervals": [
            {"interval": "[100, 200)", "slope": 0.8, "intercept": 10},
        ],
    }
    result = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},
        policy_date=datetime.date(2022, 6, 1),
        xnp=xnp,
    )
    params = result["foo"].value
    # First interval updated in 2021
    assert params.coefficients[0][0] == pytest.approx(0.9)
    # Second interval updated in 2022
    assert params.coefficients[1][0] == pytest.approx(0.8)


def test_updates_previous_on_first_date_raises_dict(xnp: ModuleType):
    """updates_previous on the initial date entry raises for dict params."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "dict",
        datetime.date(2020, 1, 1): {
            "updates_previous": True,
            "a": 1,
        },
    }
    with pytest.raises(ValueError, match="initial date entry"):
        _active_param_objects(
            orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
            policy_date=datetime.date(2020, 6, 1),
            xnp=xnp,
        )


def test_updates_previous_on_first_date_raises_piecewise(
    piecewise_spec_base, xnp: ModuleType
):
    """updates_previous on the initial date entry raises for piecewise params."""
    spec = piecewise_spec_base
    # Overwrite the only date entry to have updates_previous
    spec[datetime.date(2020, 1, 1)] = {
        "updates_previous": True,
        "intervals": [
            {"interval": "[0, 100)", "slope": 0.5, "intercept": 0},
        ],
    }
    with pytest.raises(ValueError, match="initial date entry"):
        _active_param_objects(
            orig={("spam.yaml", "foo"): spec},
            policy_date=datetime.date(2020, 6, 1),
            xnp=xnp,
        )


def test_scalar_updates_previous_raises(xnp: ModuleType):
    """updates_previous on a scalar parameter raises ValueError."""
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "scalar",
        datetime.date(2020, 1, 1): {"value": 1},
        datetime.date(2021, 1, 1): {"updates_previous": True, "value": 2},
    }
    with pytest.raises(ValueError, match="scalar parameters"):
        _active_param_objects(
            orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
            policy_date=datetime.date(2021, 6, 1),
            xnp=xnp,
        )
