"""Boundary tests for `InputData.{tree, flat, qname}` accepting scalar leaves.

Under the lazy-broadcast model, scalars stay scalars through the input
pipeline and are only materialised as arrays at the call site of a
non-vectorized function. Every `InputData.*` factory must therefore accept
scalar leaves at the `@beartype`-checked user boundary; `_canonicalize_*`
helpers and the call-time broadcast handle the rest.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ttsim import InputData
from ttsim.exceptions import InputDataError


def test_input_data_tree_accepts_scalar_only_leaves() -> None:
    """`InputData.tree` admits scalars at every leaf position."""
    InputData.tree(
        {
            "ist_selbstständig": False,
            "anzahl_kinder": 0,
            "wage_m": 1234.5,
        }
    )


def test_input_data_flat_accepts_scalar_only_leaves() -> None:
    """`InputData.flat` admits scalars at every leaf position."""
    InputData.flat(
        {
            ("ist_selbstständig",): False,
            ("anzahl_kinder",): 0,
            ("wage_m",): 1234.5,
        }
    )


def test_input_data_qname_accepts_scalar_only_leaves() -> None:
    """`InputData.qname` admits scalars at every leaf position."""
    InputData.qname(
        {
            "ist_selbstständig": False,
            "anzahl_kinder": 0,
            "wage_m": 1234.5,
        }
    )


def test_input_data_flat_accepts_single_scalar_leaf() -> None:
    """`InputData.flat({(\"x\",): False})` is admitted at the boundary."""
    InputData.flat({("ist_selbstständig",): False})


def test_input_data_qname_accepts_single_scalar_leaf() -> None:
    """`InputData.qname({\"x\": 0})` is admitted at the boundary."""
    InputData.qname({"anzahl_kinder": 0})


def test_input_data_flat_rejects_unsupported_leaf_type() -> None:
    """Strings (and other non-numeric, non-column types) are still rejected."""
    bad: Any = {("a",): "not a column"}
    with pytest.raises(InputDataError):
        InputData.flat(cast("Any", bad))


def test_input_data_qname_rejects_unsupported_leaf_type() -> None:
    """Strings (and other non-numeric, non-column types) are still rejected."""
    bad: Any = {"a": "not a column"}
    with pytest.raises(InputDataError):
        InputData.qname(cast("Any", bad))
