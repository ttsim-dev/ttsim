from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

from ttsim import policy_function
from ttsim.policy_environment import (
    ConflictingTimeDependentObjectsError,
    fail_if_multiple_ttsim_objects_are_active_at_the_same_time,
)

if TYPE_CHECKING:
    from ttsim.typing import FlatTTSIMObjectDict


@pytest.mark.parametrize(
    "date_string, expected",
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
    with pytest.raises(ValueError):

        @policy_function(start_date=date_string)
        def test_func():
            pass


def test_start_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.start_date == datetime.date(1900, 1, 1)


# End date -------------------------------------------------


@pytest.mark.parametrize(
    "date_string, expected",
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
    with pytest.raises(ValueError):

        @policy_function(end_date=date_string)
        def test_func():
            pass


def test_end_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.end_date == datetime.date(2100, 12, 31)


# Change name ----------------------------------------------


def test_dates_active_change_name_given():
    @policy_function(leaf_name="renamed_func")
    def test_func():
        pass

    assert test_func.leaf_name == "renamed_func"


def test_dates_active_change_name_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.leaf_name == "test_func"


# Empty interval -------------------------------------------


def test_dates_active_empty_interval():
    with pytest.raises(ValueError):

        @policy_function(start_date="2023-01-20", end_date="2023-01-19")
        def test_func():
            pass


# Conflicts ------------------------------------------------


def identity(x):
    return x


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree",
    [
        # Same global module, no overlapping periods.
        {
            ("a",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2023-02-01",
                end_date="2023-02-28",
                leaf_name="f",
            )(identity),
        },
        # Same submodule, no overlapping periods.
        {
            ("c", "a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("c", "b"): policy_function(
                start_date="2023-01-01",
                end_date="2023-02-28",
                leaf_name="g",
            )(identity),
        },
        # Different modules, no overlapping periods.
        {
            ("c", "f"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
            )(identity),
            ("d", "f"): policy_function(
                start_date="2023-02-01",
                end_date="2023-02-28",
            )(identity),
        },
        # Different paths, overlapping periods.
        {
            ("x", "c", "a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("y", "c", "b"): policy_function(
                start_date="2023-01-01",
                end_date="2023-02-28",
                leaf_name="g",
            )(identity),
        },
    ],
)
def test_dates_active_no_conflicts(orig_ttsim_objects_tree):
    fail_if_multiple_ttsim_objects_are_active_at_the_same_time(
        orig_ttsim_objects_tree=orig_ttsim_objects_tree
    )


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree",
    [
        # Exact overlap.
        {
            ("a",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Active period for "a" is subset of "b".
        {
            ("a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b"): policy_function(
                start_date="2021-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
        },
        # Some overlap.
        {
            ("a",): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Same as before, but defined in different modules.
        {
            ("c", "a"): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
            ("d", "b"): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Same as before, but defined in different modules without leaf name.
        {
            ("c", "f"): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
            )(identity),
            ("d", "f"): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
            )(identity),
        },
    ],
)
def test_dates_active_with_conflicts(orig_ttsim_objects_tree: FlatTTSIMObjectDict):
    with pytest.raises(ConflictingTimeDependentObjectsError):
        fail_if_multiple_ttsim_objects_are_active_at_the_same_time(
            orig_ttsim_objects_tree=orig_ttsim_objects_tree
        )
