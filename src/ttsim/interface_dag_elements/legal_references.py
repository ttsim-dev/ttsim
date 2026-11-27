"""Lookup of policy functions and parameters by their references to the law (GEP 11)."""

from __future__ import annotations

from collections.abc import Hashable

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.tt.column_objects_param_function import ParamFunction, PolicyFunction
from ttsim.tt.legal_references import (
    LegalReferenceParser,
    LegalReferences,
    NoLegalReference,
)
from ttsim.tt.param_objects import ParamObject
from ttsim.typing import PolicyEnvironment

#: The parsed references of a policy object, or the reason why there is none.
ParsedLegalReferences = tuple[Hashable, ...] | NoLegalReference


@interface_input(in_top_level_namespace=True)
def legal_references_parser() -> LegalReferenceParser:
    """The parser for references to the law.

    A user-supplied leaf like `unit_system`: a policy package passes its parser to
    ``main(legal_references_parser=...)``. Packages wrapping ``main`` (GETTSIM) default
    it to their own, so their users never name it.
    """


@interface_input(in_top_level_namespace=True)
def legal_references_query() -> str | tuple[str, ...]:
    """One or several provisions to look up, e.g., `§ 66 EStG`."""


def _parse(
    parser: LegalReferenceParser,
    qname: str,
    reference: str,
) -> Hashable:
    try:
        return parser.parse_reference(reference)
    except ValueError as e:
        raise ValueError(
            f"The reference of {qname!r} cannot be interpreted:\n\n{reference}\n\n{e}"
        ) from e


def _unparsed_references(
    qname: str,
    obj: PolicyFunction | ParamFunction | ParamObject,
) -> tuple[LegalReferences | None, LegalReferences | None]:
    """The references of a policy object and of its rounding rule."""
    try:
        refs = obj.legal_references
        rounding_refs = (
            obj.rounding_spec.legal_references
            if isinstance(obj, PolicyFunction) and obj.rounding_spec is not None
            else None
        )
    except ValueError as e:
        raise ValueError(f"The references of {qname!r} are malformed:\n\n{e}") from e
    return refs, rounding_refs


@interface_function()
def by_path(
    policy_environment: PolicyEnvironment,
    legal_references_parser: LegalReferenceParser,
) -> dict[str, ParsedLegalReferences]:
    """Map each path in the policy environment to its parsed references.

    Covers policy functions (including the references of their rounding rules),
    parameter functions, and parameters. Paths without any reference are left out.
    """
    out: dict[str, ParsedLegalReferences] = {}
    for qname, obj in dt.flatten_to_qnames(policy_environment).items():
        if not isinstance(obj, PolicyFunction | ParamFunction | ParamObject):
            continue
        refs, rounding_refs = _unparsed_references(qname=qname, obj=obj)
        if isinstance(refs, NoLegalReference) and not isinstance(rounding_refs, tuple):
            out[qname] = refs
            continue
        strings = [
            *(refs if isinstance(refs, tuple) else ()),
            *(rounding_refs if isinstance(rounding_refs, tuple) else ()),
        ]
        if not strings:
            if isinstance(rounding_refs, NoLegalReference):
                out[qname] = rounding_refs
            continue
        out[qname] = tuple(
            dict.fromkeys(
                _parse(parser=legal_references_parser, qname=qname, reference=s)
                for s in strings
            )
        )
    return out


def _matching_paths(
    query: str,
    references: dict[str, ParsedLegalReferences],
    parser: LegalReferenceParser,
) -> dict[str, tuple[Hashable, ...]]:
    parsed_query = parser.parse_query(query)
    out: dict[str, tuple[Hashable, ...]] = {}
    for qname, refs in references.items():
        if isinstance(refs, NoLegalReference):
            continue
        matching = tuple(r for r in refs if parser.matches(parsed_query, r))
        if matching:
            out[qname] = matching
    return out


@interface_function()
def by_query(
    by_path: dict[str, ParsedLegalReferences],
    legal_references_query: str | tuple[str, ...],
    legal_references_parser: LegalReferenceParser,
) -> dict[str, tuple[Hashable, ...]] | dict[str, dict[str, tuple[Hashable, ...]]]:
    """The paths whose references match the query, with the matching references.

    If the query is a tuple of queries, maps each query to its result.
    """
    if isinstance(legal_references_query, str):
        return _matching_paths(
            query=legal_references_query,
            references=by_path,
            parser=legal_references_parser,
        )
    return {
        query: _matching_paths(
            query=query,
            references=by_path,
            parser=legal_references_parser,
        )
        for query in legal_references_query
    }
