import pytest

from ttsim.tt.param_objects import convert_sparse_ints_to_consecutive_int_lookup_table


@pytest.mark.parametrize(
    (
        "raw",
        "expected_result",
    ),
    [
        (
            {1: 1, 3: 3, "min_int_in_table": 0, "max_int_in_table": 5},
            {0: 1, 1: 1, 2: 1, 3: 3, 4: 3, 5: 3},
        ),
        (
            {1: 1, 3: 0, "min_int_in_table": 0, "max_int_in_table": 5},
            {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0},
        ),
    ],
)
def test_convert_sparse_ints_to_consecutive_int_lookup_table(raw, expected_result, xnp):
    result = convert_sparse_ints_to_consecutive_int_lookup_table(raw, xnp)
    for k, v in expected_result.items():
        assert result.look_up(k) == v
