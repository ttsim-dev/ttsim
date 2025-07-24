from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import (
        OrderedQNames,
        QNameData,
        SpecEnvWithProcessedParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def columns(
    labels__root_nodes: UnorderedQNames,
    processed_data: QNameData,
    specialized_environment__tax_transfer_function: Callable[[QNameData], QNameData],
) -> QNameData:
    return specialized_environment__tax_transfer_function(
        {k: v for k, v in processed_data.items() if k in labels__root_nodes},
    )


@interface_function()
def params(
    labels__param_targets: OrderedQNames,
    specialized_environment__with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars,  # noqa: E501
) -> QNameData:
    return {
        pt: specialized_environment__with_processed_params_and_scalars[pt]
        for pt in labels__param_targets
    }


@interface_function()
def from_input_data(
    labels__input_data_targets: OrderedQNames,
    processed_data: QNameData,
) -> QNameData:
    return {ot: processed_data[ot] for ot in labels__input_data_targets}


@interface_function()
def combined(
    columns: QNameData,
    params: QNameData,
    from_input_data: QNameData,
) -> QNameData:
    return {
        **columns,
        **params,
        **from_input_data,
    }
