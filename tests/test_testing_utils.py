"""Tests for the helpers a policy package's own test suite builds on."""

from __future__ import annotations

import datetime
from typing import Any

from tests.test_unit_fixtures import UNIT_SYSTEM
from ttsim.testing_utils import get_policy_date_partition
from ttsim.tt import TTSIMUnit, policy_function


@policy_function(
    leaf_name="amount_m",
    start_date="2020-01-01",
    end_date="2029-12-31",
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def amount_m(rate_m: float) -> float:
    return rate_m


@policy_function(
    leaf_name="early_amount_m",
    start_date="2010-01-01",
    end_date="2029-12-31",
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def early_amount_m(rate_m: float) -> float:
    """Active across the test unit system's 2020 statutory-currency transition."""
    return rate_m


@policy_function(
    leaf_name="short_amount_m",
    start_date="2020-01-01",
    end_date="2024-12-31",
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def short_amount_m(rate_m: float) -> float:
    """End while another policy function remains active."""
    return rate_m


def _orig_objects(
    param_spec: dict[Any, Any],
    function: Any = amount_m,
) -> dict[str, Any]:
    return {
        "column_objects_and_param_functions": {("amount_m",): function},
        "param_specs": {("rate_m",): param_spec},
    }


def test_policy_date_partition_covers_function_start_and_end():
    """An end boundary is included only while the package remains supported."""
    orig_objects = _orig_objects({"name": "Rate", datetime.date(2020, 1, 1): 1})
    orig_objects["column_objects_and_param_functions"][("short_amount_m",)] = (
        short_amount_m
    )

    assert get_policy_date_partition(orig_policy_objects=orig_objects) == [
        datetime.date(2020, 1, 1),
        datetime.date(2025, 1, 1),
    ]


def test_policy_date_partition_includes_a_parameter_only_change():
    """A parameter changing mid-interval starts a regime of its own — every function
    stays active, so a partition built from function dates alone never assembles the
    new parameter regime (GEP 10)."""
    assert datetime.date(2022, 7, 1) in get_policy_date_partition(
        orig_policy_objects=_orig_objects(
            {
                "name": "Rate",
                datetime.date(2020, 1, 1): 1,
                datetime.date(2022, 7, 1): 2,
            }
        )
    )


def test_policy_date_partition_includes_a_statutory_currency_change():
    """A statutory-currency transition changes what the environment's amounts are
    denominated in, so it starts a regime too — even where no function or parameter
    boundary falls on it (GEP 10)."""
    assert datetime.date(2020, 1, 1) in get_policy_date_partition(
        orig_policy_objects=_orig_objects({"name": "Rate"}, function=early_amount_m),
        unit_system=UNIT_SYSTEM,
    )


def test_policy_date_partition_drops_boundaries_outside_the_date_domain():
    """A boundary before the package's earliest start date names no environment, so
    it is not a regime to validate — the year-1 start of a statutory currency is the
    common case."""
    assert datetime.date(1, 1, 1) not in get_policy_date_partition(
        orig_policy_objects=_orig_objects({"name": "Rate"}),
        unit_system=UNIT_SYSTEM,
    )
