"""Input columns."""

from ttsim import policy_input


@policy_input()
def gross_wage_y() -> float:
    """Annual gross wage."""
