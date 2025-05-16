"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def baujahr_immobilie_hh() -> int:
    """Year of construction of the household dwelling."""


@policy_input()
def bewohnt_eigentum_hh() -> bool:
    """Owner-occupied housing."""


@policy_input()
def bruttokaltmiete_m_hh() -> float:
    """Rent expenses excluding utilities."""


@policy_input()
def heizkosten_m_hh() -> float:
    """Heating expenses."""


@policy_input()
def wohnfläche_hh() -> float:
    """Size of household dwelling in square meters."""
