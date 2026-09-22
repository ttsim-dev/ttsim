import pint
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintViolation

from ttsim._beartype_conf import INTERNAL_CONF


def _registry_identity(registry: pint.UnitRegistry) -> pint.UnitRegistry:
    return registry


def test_internal_conf_accepts_unit_registry() -> None:
    """A `pint.UnitRegistry` annotation is checkable on every supported Python."""
    registry = pint.UnitRegistry()
    assert beartype(conf=INTERNAL_CONF)(_registry_identity)(registry) is registry


def test_internal_conf_rejects_non_registry_for_unit_registry() -> None:
    """The `pint.UnitRegistry` check still rejects other objects."""
    checked = beartype(conf=INTERNAL_CONF)(_registry_identity)
    with pytest.raises(BeartypeCallHintViolation):
        checked("not-a-registry")  # ty: ignore[invalid-argument-type]
