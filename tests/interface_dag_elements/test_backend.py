"""Tests for `backend` module."""

from __future__ import annotations

import numpy
import pytest

from ttsim.interface_dag_elements.backend import dnp, xnp

# =============================================================================
# xnp() tests
# =============================================================================


def test_xnp_returns_numpy_for_numpy_backend():
    """Test xnp('numpy') returns the numpy module."""
    result = xnp("numpy")
    assert result is numpy


@pytest.mark.skipif_numpy
def test_xnp_returns_jax_numpy_for_jax_backend():
    """Test xnp('jax') returns jax.numpy module."""
    import jax  # noqa: PLC0415

    result = xnp("jax")
    assert result is jax.numpy


def test_xnp_invalid_backend_raises_value_error():
    """Test xnp raises ValueError for invalid backend string."""
    with pytest.raises(ValueError, match=r"Unsupported backend.*invalid"):
        xnp("invalid")


def test_xnp_case_sensitive():
    """Test xnp is case-sensitive (e.g. 'NumPy' raises ValueError)."""
    with pytest.raises(ValueError, match=r"Unsupported backend.*NumPy"):
        xnp("NumPy")


def test_xnp_empty_string_raises():
    """Test xnp raises ValueError for empty string."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        xnp("")


# =============================================================================
# dnp() tests
# =============================================================================


def test_dnp_returns_numpy_for_numpy_backend():
    """Test dnp('numpy') returns the numpy module."""
    result = dnp("numpy")
    assert result is numpy


def test_dnp_returns_numpy_for_jax_backend():
    """Test dnp('jax') currently returns numpy (jax_datetime not implemented)."""
    # Currently jax backend also returns numpy for datetime operations
    result = dnp("jax")
    assert result is numpy
