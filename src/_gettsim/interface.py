from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from _gettsim.config import GETTSIM_ROOT
from ttsim import main
from ttsim.interface_dag_elements.data_converters import (
    mapped_dataframe_to_nested_data,
    nested_data_to_df_with_mapped_columns,
)
from ttsim.interface_dag_elements.shared import to_datetime

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import NestedInputs, NestedStrings


def oss(
    date: str,
    inputs_df: pd.DataFrame,
    inputs_tree_to_inputs_df_columns: NestedInputs,
    targets__tree: NestedStrings,
) -> pd.DataFrame:
    """One-stop-shop for computing taxes and transfers.

    Args:
        date:
            The date to compute taxes and transfers for. The date determines the policy
            environment for which the taxes and transfers are computed.
        inputs_df:
            The DataFrame containing the data.
        inputs_tree_to_inputs_df_columns:
            A tree that has the inputs required by GETTSIM as the path (sequence of
            keys) and maps them to the data provided by the user. The leaves of the tree
            are strings that reference column names in *inputs_df* or constants.
        targets__tree:
            A tree that has the desired targets as the path (sequence of keys) and maps
            them to the data columns the user would like to have.

    Returns
    -------
        A DataFrame with the results.

    Examples
    --------
    >>> inputs_df = pd.DataFrame(
    ...     {
    ...         "age": [25, 45, 3, 65],
    ...         "wage": [950, 950, 0, 950],
    ...         "id": [0, 1, 2, 3],
    ...         "hh_id": [0, 1, 1, 2],
    ...         "mother_id": [-1, -1, 1, -1],
    ...         "has_kids": [False, True, False, True],
    ...     }
    ... )
    >>> inputs_map = {
    ...     "p_id": "id",
    ...     "hh_id": "hh_id",
    ...     "alter": "age",
    ...     "familie":{
    ...         "p_id_elternteil_1": "mother_id",
    ...         "p_id_elternteil_2": -1,
    ...     },
    ...     "einkommensteuer": {
    ...         "einkünfte": {
    ...             "aus_nichtselbstständiger_arbeit": {"bruttolohn_m": "wage"},
    ...             "ist_selbstständig": False,
    ...             "aus_selbstständiger_arbeit": {"betrag_m": 0.0},
    ...         }
    ...     },
    ...     "sozialversicherung": {
    ...         "pflege": {
    ...             "beitrag": {
    ...                 "hat_kinder": "has_kids",
    ...             }
    ...         },
    ...         "kranken": {
    ...             "beitrag":{
    ...                 "bemessungsgrundlage_rente_m": 0.0,
    ...                 "privat_versichert": False
    ...             }
    ...         }
    ...     },
    ... }
    >>> targets_map={
    ...        "sozialversicherung": {
    ...            "pflege": {
    ...                "beitrag": {
    ...                    "betrag_versicherter_m": "ltci_contrib",
    ...                }
    ...            }
    ...        }
    ...    }
    >>> oss(
    ...     date="2025-01-01",
    ...     inputs_df=inputs_df,
    ...     inputs_tree_to_inputs_df_columns=inputs_map,
    ...     targets__tree=targets_map,
    ... )
       ltci_contrib
    0         14.72
    1          9.82
    2          0.00
    3          9.82
    """
    input_data__tree = mapped_dataframe_to_nested_data(
        mapper=inputs_tree_to_inputs_df_columns,
        df=inputs_df,
        xnp=numpy,
    )
    nested_result = main(
        inputs={
            "date": to_datetime(date),
            "orig_policy_objects__root": GETTSIM_ROOT,
            "input_data__tree": input_data__tree,
            "targets__tree": targets__tree,
            "rounding": True,
        },
        targets=["results__tree"],
    )["results__tree"]
    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=nested_result,
        nested_outputs_df_column_names=targets__tree,
        data_with_p_id=input_data__tree,
    )
