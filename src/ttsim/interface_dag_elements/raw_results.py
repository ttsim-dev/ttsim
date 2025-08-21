from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import (
        FlatData,
        OrderedQNames,
        QNameData,
        SpecEnvWithProcessedParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def columns(
    labels__root_nodes: UnorderedQNames,
    processed_data: QNameData,
    tt_function: Callable[[QNameData], QNameData],
) -> QNameData:
    """The raw results of the TT function that have been requested as targets.

    Arrays are sorted according to the internal sort order.

    """
    return tt_function(
        {k: v for k, v in processed_data.items() if k in labels__root_nodes},
    )


@interface_function()
def from_input_data(
    labels__input_data_targets: OrderedQNames,
    input_data__flat: FlatData,
) -> QNameData:
    """The input data columns that have been requested as targets.

    Arrays are sorted as they are in the input data.

    """
    return {
        target: input_data__flat[dt.tree_path_from_qname(target)]
        for target in labels__input_data_targets
    }


@interface_function()
def params(
    labels__param_targets: OrderedQNames,
    specialized_environment__with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars,  # noqa: E501
) -> QNameData:
    """The parameters that have been requested as targets.

    Possibly includes outputs of param_functions.
    """
    return {
        pt: specialized_environment__with_processed_params_and_scalars[pt]
        for pt in labels__param_targets
    }
