"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def gemeinsam_veranlagt() -> bool:
    """Taxes are filed jointly."""
