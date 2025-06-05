from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import NestedData, QualNameData


def processed__data(
    input_data__tree: NestedData,
) -> QualNameData:
    """Process the data for use in the taxes and transfers function.

    This is where the conversion of p_ids will happen.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns:
        A DataFrame.
    """
    return dt.flatten_to_tree_paths(input_data__tree)
