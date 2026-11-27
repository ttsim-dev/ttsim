from __future__ import annotations

import pytest

from ttsim.tt import RoundingSpec, ScalarParam, TTSIMUnit, policy_function
from ttsim.tt.legal_references import (
    NoLegalReference,
    legal_references_from_docstring,
    legal_references_from_spec,
)


@pytest.mark.parametrize(
    ("docstring", "expected"),
    [
        (None, None),
        ("Some function.", None),
        (
            "Some function.\n\nReference: § 1 Abs. 2 XG idF G. v. 1",
            ("§ 1 Abs. 2 XG idF G. v. 1",),
        ),
        (
            "Some function.\n\nReference: § 1 Abs. 2 XG\n    idF G. v. 1\n\nMore text.",
            ("§ 1 Abs. 2 XG idF G. v. 1",),
        ),
        (
            "F.\n\nReference:\n    - § 1 XG idF A\n    - § 2 XG\n      idF B",
            ("§ 1 XG idF A", "§ 2 XG idF B"),
        ),
        ("Reference: none (technical helper)", NoLegalReference("technical helper")),
        ("Reference: none(helper (internal))", NoLegalReference("helper (internal)")),
    ],
)
def test_legal_references_from_docstring(docstring, expected):
    assert legal_references_from_docstring(docstring) == expected


@pytest.mark.parametrize(
    "docstring",
    [
        "Reference:\n    § 1 XG",
        "Reference:\n\nSome text.",
    ],
)
def test_legal_references_from_docstring_fails_for_malformed_sections(docstring):
    with pytest.raises(ValueError, match="Reference:"):
        legal_references_from_docstring(docstring)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, None),
        ("§ 1 XG idF A", ("§ 1 XG idF A",)),
        (["§ 1 XG idF A", "§ 2 XG idF  B"], ("§ 1 XG idF A", "§ 2 XG idF B")),
        ("none (no longer applies)", NoLegalReference("no longer applies")),
    ],
)
def test_legal_references_from_spec(spec, expected):
    assert legal_references_from_spec(spec) == expected


def test_policy_function_has_legal_references():
    @policy_function(
        unit=TTSIMUnit.DIMENSIONLESS,
        rounding_spec=RoundingSpec(base=1, direction="up", reference="§ 3 XG idF C"),
    )
    def f(x: float) -> float:
        """Docstring.

        Reference: § 1 XG idF A
        """
        return x

    assert f.legal_references == ("§ 1 XG idF A",)
    assert f.rounding_spec is not None
    assert f.rounding_spec.legal_references == ("§ 3 XG idF C",)


def test_param_object_has_legal_references():
    p = ScalarParam(value=1, reference=["§ 1 XG idF A", "§ 2 XG idF B"])
    assert p.legal_references == ("§ 1 XG idF A", "§ 2 XG idF B")


def test_rounding_spec_accepts_several_references():
    spec = RoundingSpec(
        base=1, direction="up", reference=("§ 1 XG idF A", "§ 2 XG idF B")
    )
    assert spec.legal_references == ("§ 1 XG idF A", "§ 2 XG idF B")
