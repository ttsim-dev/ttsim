from __future__ import annotations

import pytest

from ttsim.interface_dag_elements.legal_references import by_path, by_query
from ttsim.tt import RoundingSpec, ScalarParam, TTSIMUnit, policy_function, policy_input
from ttsim.tt.legal_references import LegalReferenceParser, NoLegalReference


class ToyParser:
    """References are '§ <n> <law> idF <act>'; queries are '§<n> <law>'."""

    def parse_reference(self, reference: str) -> tuple[str, str, str]:
        provision, _, act = reference.partition(" idF ")
        if not act:
            raise ValueError("missing act")
        paragraph, law = provision.removeprefix("§ ").split(" ", 1)
        return (paragraph, law, act)

    def parse_query(self, query: str) -> tuple[str, str]:
        paragraph, law = query.replace("§", "").split()
        return (paragraph, law.upper())

    def matches(self, query: tuple[str, str], reference: tuple[str, str, str]) -> bool:
        return query == (reference[0], reference[1].upper())


@policy_function(unit=TTSIMUnit.DIMENSIONLESS)
def f(x: float) -> float:
    """F.

    Reference:
        - § 1 XG idF A
        - § 2 XG idF B
    """
    return x


@policy_function(
    unit=TTSIMUnit.DIMENSIONLESS,
    rounding_spec=RoundingSpec(base=1, direction="up", reference="§ 3 XG idF C"),
)
def g(x: float) -> float:
    """G.

    Reference: none (technical helper)
    """
    return x


@policy_function(unit=TTSIMUnit.DIMENSIONLESS)
def h(x: float) -> float:
    """H.

    Reference: none (technical helper)
    """
    return x


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def x() -> float:
    pass


POLICY_ENVIRONMENT = {
    "x": x,
    "n": {"f": f, "g": g, "h": h},
    "p": ScalarParam(value=1, reference="§ 2 YG idF D"),
}


def test_toy_parser_satisfies_protocol():
    assert isinstance(ToyParser(), LegalReferenceParser)


def test_by_path():
    assert by_path(
        policy_environment=POLICY_ENVIRONMENT, legal_references_parser=ToyParser()
    ) == {
        "n__f": (("1", "XG", "A"), ("2", "XG", "B")),
        "n__g": (("3", "XG", "C"),),
        "n__h": NoLegalReference("technical helper"),
        "p": (("2", "YG", "D"),),
    }


def test_by_path_fails_for_malformed_reference():
    with pytest.raises(ValueError, match="'p'"):
        by_path(
            policy_environment={"p": ScalarParam(value=1, reference="§ 2 YG")},
            legal_references_parser=ToyParser(),
        )


@policy_function(unit=TTSIMUnit.DIMENSIONLESS)
def malformed(x: float) -> float:
    """Malformed.

    Reference:
    § 1 XG idF A
    """
    return x


def test_by_path_fails_for_malformed_reference_section():
    with pytest.raises(ValueError, match="'n__malformed'"):
        by_path(
            policy_environment={"x": x, "n": {"malformed": malformed}},
            legal_references_parser=ToyParser(),
        )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("§2 xg", {"n__f": (("2", "XG", "B"),)}),
        ("§ 9 XG", {}),
        (
            ("§ 2 xg", "§ 2 yg"),
            {
                "§ 2 xg": {"n__f": (("2", "XG", "B"),)},
                "§ 2 yg": {"p": (("2", "YG", "D"),)},
            },
        ),
    ],
)
def test_by_query(query, expected):
    parser = ToyParser()
    assert (
        by_query(
            by_path=by_path(
                policy_environment=POLICY_ENVIRONMENT, legal_references_parser=parser
            ),
            legal_references_query=query,
            legal_references_parser=parser,
        )
        == expected
    )
