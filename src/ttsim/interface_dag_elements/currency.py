from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import pint

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    strip_input_quantity_at_boundary,
    ttsim_unit_has_currency,
)
from ttsim.typing import SpecEnvWithoutTreeLogicAndWithDerivedFunctions

if TYPE_CHECKING:
    from ttsim.typing import FlatData


@dataclass(frozen=True)
class CurrencyConversion:
    """The scalar factor crossing one currency into another, and where it applies."""

    factor: float
    """Multiplier taking a magnitude from the source into the target currency."""
    qnames: frozenset[str]
    """The qualified names the factor applies to — those whose declared unit
    carries a currency component."""

    @classmethod
    def between(
        cls,
        *,
        source_currency: str,
        target_currency: str,
        qnames: Iterable[str],
        specialized_environment: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        unit_system: UnitSystem,
    ) -> CurrencyConversion:
        """The conversion crossing `source_currency` into `target_currency`.

        Args:
            source_currency: The currency the values are denominated in.
            target_currency: The currency the values should end up in.
            qnames: The candidate qualified names; those whose declared unit
                carries a currency component end up in `qnames`.
            specialized_environment: Supplies each qname's declared unit.
            unit_system: Supplies the conversion factor.

        Returns:
            The conversion; the identity (factor `1.0`, no qnames) when source
            and target currency coincide.
        """
        if source_currency == target_currency:
            return cls(factor=1.0, qnames=frozenset())
        return cls(
            factor=unit_system.currency_conversion_factor(
                source_currency=source_currency,
                target_currency=target_currency,
            ),
            qnames=_qnames_with_currency_declarations(
                qnames=qnames,
                specialized_environment=specialized_environment,
            ),
        )

    def apply(
        self,
        value: Any,  # noqa: ANN401 (a column array or an input scalar)
        qname: str,
    ) -> Any:  # noqa: ANN401
        """`value` denominated in the target currency.

        Returned unchanged when `qname` declares no currency, or when the value
        is an object array — those carry `pd.NA` and must not be multiplied; the
        missing-value fail-if surfaces them downstream.
        """
        if qname not in self.qnames:
            return value
        dtype = getattr(value, "dtype", None)
        if dtype is not None and dtype.kind == "O":
            return value
        return value * self.factor


@interface_function(leaf_name="data_currency", in_top_level_namespace=True)
def data_currency(unit_system: UnitSystem) -> str:
    """The currency the user's data arrives in and results are returned in.

    Defaults to the policy system's base currency. Override via
    ``main(data_currency=...)`` with another of the system's currencies.
    """
    return unit_system.base_currency


@interface_function(leaf_name="computation_currency", in_top_level_namespace=True)
def computation_currency(policy_date: datetime.date, unit_system: UnitSystem) -> str:
    """The currency the computation runs in — the policy date's statutory currency.

    Read off the dated mapping the policy system declares.
    """
    return unit_system.statutory_currency_for_date(policy_date)


@interface_function(in_top_level_namespace=True)
def input_data_in_computation_currency(
    input_data__flat: FlatData,
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    data_currency: str,
    computation_currency: str,
    unit_system: UnitSystem,
) -> FlatData:
    """The input data with every value a bare magnitude in the computation currency.

    Two currency crossings happen here, and nowhere else on the way in (GEP 10):

    - a column the user pint-tagged is converted into the data currency and its
      tag stripped, so the tag overrides the blanket "untagged data is in the
      data currency" assumption for that column alone;
    - every currency-denominated value is then crossed from the data currency
      into the computation currency by one scalar factor.

    `p_id` is an identifier, never currency-denominated, and passes through
    untouched. The factor is elementwise, so applying it here — before the data
    are sorted and reindexed in :func:`processed_data` — is equivalent.
    """
    registry: pint.UnitRegistry = unit_system.registry
    stripped = {
        path: (
            strip_input_quantity_at_boundary(
                quantity=value,
                data_currency=data_currency,
                registry=registry,
                column_label=dt.qname_from_tree_path(path),
            )
            if isinstance(value, pint.Quantity)
            else value
        )
        for path, value in input_data__flat.items()
    }
    conversion = CurrencyConversion.between(
        source_currency=data_currency,
        target_currency=computation_currency,
        qnames=[
            dt.qname_from_tree_path(path) for path in stripped if path != ("p_id",)
        ],
        specialized_environment=(
            specialized_environment__without_tree_logic_and_with_derived_functions
        ),
        unit_system=unit_system,
    )
    return {
        path: (
            value
            if path == ("p_id",)
            else conversion.apply(value=value, qname=dt.qname_from_tree_path(path))
        )
        for path, value in stripped.items()
    }


def _qnames_with_currency_declarations(
    qnames: Iterable[str],
    specialized_environment: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> frozenset[str]:
    """The subset of `qnames` whose declared unit carries a currency component."""
    return frozenset(
        qname
        for qname in qnames
        if isinstance(
            token := getattr(specialized_environment.get(qname), "unit", UNSET_UNIT),
            CompositeUnit,
        )
        and ttsim_unit_has_currency(token)
    )
