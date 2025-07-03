from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import networkx as nx
    import pandas as pd

    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        NestedData,
        NestedStrings,
        QNameData,
        QNameSpecializedEnvironment0,
        QNameSpecializedEnvironment1,
        QNameSpecializedEnvironment2,
        QNameStrings,
    )


@dataclass(frozen=True)
class MainArg:
    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class OrigPolicyObjects(MainArg):
    root: Path | None = None
    column_objects_and_param_functions: FlatColumnObjectsParamFunctions | None = None
    param_specs: FlatOrigParamSpecs | None = None


@dataclass(frozen=True)
class RawResults(MainArg):
    columns: QNameData | None = None
    params: QNameData | None = None
    from_input_data: QNameData | None = None
    combined: QNameData | None = None


@dataclass(frozen=True)
class Results(MainArg):
    df_with_mapper: pd.DataFrame | None = None
    df_with_nested_columns: pd.DataFrame | None = None
    tree: NestedData | None = None


@dataclass(frozen=True)
class Targets(MainArg):
    qname: QNameStrings | None = None
    tree: NestedStrings | None = None


@dataclass(frozen=True)
class SpecializedEnvironment(MainArg):
    without_tree_logic_and_with_derived_functions: (
        QNameSpecializedEnvironment0 | None
    ) = None
    with_processed_params_and_scalars: QNameSpecializedEnvironment1 | None = None
    with_partialled_params_and_scalars: QNameSpecializedEnvironment2 | None = None
    tax_transfer_dag: nx.DiGraph | None = None
    tax_transfer_function: Callable[[QNameData], QNameData] | None = None
