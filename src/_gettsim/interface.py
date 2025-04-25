import pandas as pd

from _gettsim.config import FOREIGN_KEYS, RESOURCE_DIR, SUPPORTED_GROUPINGS
from ttsim import (
    compute_taxes_and_transfers,
    create_data_tree,
    set_up_policy_environment,
)
from ttsim.typing import NestedDataDict, NestedInputToSeriesNameDict


def quickrun(
    date: str,
    df: pd.DataFrame,
    input_tree_to_column_name_mapping: NestedInputToSeriesNameDict,
    targets_tree: NestedDataDict,
) -> NestedDataDict:
    """Quickly compute taxes and transfers for a given date."""
    data_tree = create_data_tree(df, input_tree_to_column_name_mapping)
    policy_environment = set_up_policy_environment(
        date=date,
        resource_dir=RESOURCE_DIR,
    )
    return compute_taxes_and_transfers(
        data_tree=data_tree,
        environment=policy_environment,
        targets_tree=targets_tree,
        foreign_keys=FOREIGN_KEYS,
        supported_groupings=SUPPORTED_GROUPINGS,
        rounding=True,
        debug=False,
    )
