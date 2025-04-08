"""Input columns."""

from ttsim import policy_input


@policy_input()
def baujahr_immobilie_hh() -> int:
    """Year of construction of the household dwelling."""
    return 0


@policy_input()
def bewohnt_eigentum_hh() -> bool:
    """Owner-occupied housing."""
    return False


@policy_input()
def bruttokaltmiete_m_hh() -> float:
    """Rent expenses excluding utilities."""
    return 0


@policy_input()
def heizkosten_m_hh() -> float:
    """Heating expenses."""
    return 0


@policy_input()
def wohnfläche_hh() -> float:
    """Size of household dwelling in square meters."""
    return 0
