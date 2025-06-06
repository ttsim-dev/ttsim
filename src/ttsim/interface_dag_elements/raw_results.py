from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.interface_dag_elements.typing import (
        OrderedQNames,
        QNameCombinedEnvironment1,
        QNameData,
        UnorderedQNames,
    )


def raw_results__columns(
    names__root_nodes: UnorderedQNames,
    processed_data: QNameData,
    tax_transfer_function: Callable[[QNameData], QNameData],
) -> QNameData:
    return tax_transfer_function(
        {k: v for k, v in processed_data.items() if k in names__root_nodes}
    )


def raw_results__params(
    names__target_params: OrderedQNames,
    environment_with_data__with_processed_params_and_scalars: QNameCombinedEnvironment1,
) -> QNameData:
    return {
        pt: environment_with_data__with_processed_params_and_scalars[pt]
        for pt in names__target_params
    }


def raw_results__from_input_data(
    names__targets_from_input_data: OrderedQNames,
    processed_data: QNameData,
) -> QNameData:
    return {ot: processed_data[ot] for ot in names__targets_from_input_data}


def raw_results__combined(
    raw_results__columns: QNameData,
    raw_results__params: QNameData,
    raw_results__from_input_data: QNameData,
) -> QNameData:
    return {
        **raw_results__columns,
        **raw_results__params,
        **raw_results__from_input_data,
    }
