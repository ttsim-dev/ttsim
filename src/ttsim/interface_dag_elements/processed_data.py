from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.config import numpy_or_jax as np

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import NestedData, QNameData


def processed_data(
    input_data__tree: NestedData,
) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    This is where the conversion of p_ids will happen.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns:
        A DataFrame.
    """
    return {
        k: np.asarray(v) for k, v in dt.flatten_to_qual_names(input_data__tree).items()
    }
