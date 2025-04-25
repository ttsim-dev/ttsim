import pandas as pd

from _gettsim.config import RESOURCE_DIR, SUPPORTED_GROUPINGS
from ttsim import (
    compute_taxes_and_transfers,
    create_data_tree,
    set_up_policy_environment,
)
from ttsim.typing import NestedDataDict, NestedInputToSeriesNameDict


def quickrun(
    date: str,
    input_tree_to_column_map: NestedInputToSeriesNameDict,
    targets_tree: NestedDataDict,
    df: pd.DataFrame | None = None,
) -> NestedDataDict:
    """Compute taxes and transfers.

    Args:
        date:
            The date to compute taxes and transfers for. The date determines the policy
            environment for which the taxes and transfers are computed.
        input_tree_to_column_map:
            A nested dictionary that maps GETTSIM's expected input structure to the data
            provided by the user. Keys are strings that provide a path to an input.

            Values can be:

            - Strings that reference column names in the DataFrame (when df is provided)
            - pandas Series objects
            - Numeric or boolean values (which will be used directly if df is None, or
              broadcast to match the DataFrame length if df is provided)
        targets_tree:
            The targets tree.
        df:
            The DataFrame containing the data.


    Examples:
    --------
    >>> input_tree_to_column_map = {
    ...     "einkommensteuer": {
    ...         "gemeinsam_veranlagt": "joint_taxation",
    ...         "einkünfte": {
    ...             "aus_nichtselbstständiger_arbeit": {
    ...                 "bruttolohn_m": "gross_wage_m",
    ...             },
    ...         },
    ...     },
    ...     "alter": 30,
    ...     "geburtsjahr": pd.Series([1990, 1990, 1990]),
    ...     "p_id": pd.Series([0, 1, 2]),
    ... }
    >>> df = pd.DataFrame(
    ...     {
    ...         "gross_wage_m": [1000, 2000, 3000],
    ...         "joint_taxation": [True, True, False],
    ...     }
    ... )
    >>> quickrun(
    ...     date="2024-01-01",
    ...     input_tree_to_column_map=input_tree_to_column_map,
    ...     targets_tree=targets_tree,
    ...     df=df,
    ... )
    """
    data_tree = create_data_tree(
        input_tree_to_column_map=input_tree_to_column_map,
        df=df,
    )
    policy_environment = set_up_policy_environment(
        date=date,
        resource_dir=RESOURCE_DIR,
    )
    return compute_taxes_and_transfers(
        data_tree=data_tree,
        environment=policy_environment,
        targets_tree=targets_tree,
        supported_groupings=SUPPORTED_GROUPINGS,
        rounding=True,
        debug=False,
        jit=False,
    )
