from __future__ import annotations

from typing import TYPE_CHECKING

from _gettsim.config import GETTSIM_ROOT
from ttsim import (
    compute_taxes_and_transfers,
    create_data_tree_from_df,
    set_up_policy_environment,
)

if TYPE_CHECKING:
    import pandas as pd
    from dags.tree.typing import NestedTargetDict

    from ttsim.typing import NestedDataDict, NestedInputsPathsToDfColumns


def oss(
    date: str,
    df: pd.DataFrame,
    inputs_tree_to_df_columns: NestedInputsPathsToDfColumns,
    targets_tree: NestedTargetDict,
) -> NestedDataDict:
    """One-stop-shop for computing taxes and transfers.

    Args:
        date:
            The date to compute taxes and transfers for. The date determines the policy
            environment for which the taxes and transfers are computed.
        df:
            The DataFrame containing the data.
        inputs_tree_to_df_columns:
            A nested dictionary that maps GETTSIM's expected input structure to the data
            provided by the user. Keys are strings that provide a path to an input.

            Values can be:
            - Strings that reference column names in the DataFrame.
            - Numeric or boolean values (which will be broadcasted to match the length
              of the DataFrame).
        targets_tree:
            The targets tree.


    Examples:
    --------
    >>> inputs_tree_to_df_columns = {
    ...     "einkommensteuer": {
    ...         "gemeinsam_veranlagt": "joint_taxation",
    ...         "einkünfte": {
    ...             "aus_nichtselbstständiger_arbeit": {
    ...                 "bruttolohn_m": "gross_wage_m",
    ...             },
    ...         },
    ...     },
    ...     "alter": 30,
    ...     "p_id": "p_id",
    ... }
    >>> df = pd.DataFrame(
    ...     {
    ...         "gross_wage_m": [1000, 2000, 3000],
    ...         "joint_taxation": [True, True, False],
    ...         "p_id": [0, 1, 2],
    ...     }
    ... )
    >>> oss(
    ...     date="2024-01-01",
    ...     inputs_tree_to_df_columns=inputs_tree_to_df_columns,
    ...     targets_tree=targets_tree,
    ...     df=df,
    ... )
    """
    data_tree = create_data_tree_from_df(
        inputs_tree_to_df_columns=inputs_tree_to_df_columns,
        df=df,
    )
    policy_environment = set_up_policy_environment(
        date=date,
        root=GETTSIM_ROOT,
    )
    return compute_taxes_and_transfers(
        data_tree=data_tree,
        environment=policy_environment,
        targets_tree=targets_tree,
        rounding=True,
        debug=False,
        jit=False,
    )
