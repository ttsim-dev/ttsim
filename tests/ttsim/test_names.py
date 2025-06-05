from __future__ import annotations

import pytest

from ttsim import (
    names__grouping_levels,
    names__top_level_namespace,
    policy_function,
    policy_input,
)


def identity(x: int) -> int:
    return x


@policy_input()
def fam_id() -> int:
    pass


@pytest.mark.parametrize(
    (
        "policy_environment",
        "expected",
    ),
    [
        (
            {
                "foo_m": policy_function(leaf_name="foo_m")(identity),
                "fam_id": fam_id,
            },
            {"foo_m", "foo_y", "foo_m_fam", "foo_y_fam"},
        ),
        (
            {
                "foo": policy_function(leaf_name="foo")(identity),
                "fam_id": fam_id,
            },
            {"foo", "foo_fam"},
        ),
    ],
)
def test_get_top_level_namespace(policy_environment, expected):
    grouping_levels = names__grouping_levels(policy_environment=policy_environment)
    result = names__top_level_namespace(
        policy_environment=policy_environment,
        names__grouping_levels=grouping_levels,
    )
    assert all(name in result for name in expected)
