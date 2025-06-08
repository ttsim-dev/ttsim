from __future__ import annotations

import pytest

from ttsim.interface_dag_elements.names import grouping_levels, top_level_namespace
from ttsim.tt_dag_elements import policy_function, policy_input


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
    result = top_level_namespace(
        policy_environment=policy_environment,
        grouping_levels=grouping_levels(policy_environment),
    )
    assert all(name in result for name in expected)
