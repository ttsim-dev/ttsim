from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.typing import (
    FlatData,
    IntColumn,
    OrderedQNames,
    QNameData,
    QNameResults,
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

    Arrays are sorted according to the internal sort order. Endogenously- computed
    `p_id_*` columns hold internal indices; see `columns_with_remapped_ids` for the
    version with user-space `p_id` values.
    """
    return tt_function(
        {k: v for k, v in processed_data.items() if k in labels__root_nodes},
    )


@interface_function()
def columns_with_remapped_ids(
    columns: QNameData,
    input_data__flat: FlatData,
    input_data__sort_indices: IntColumn,
    xnp: ModuleType,
) -> QNameData:
    """Raw results with endogenous `p_id_*` columns remapped to user-space `p_id`s.

    Arrays are sorted according to the internal sort order. Endogenously-
    computed `p_id_*` columns hold internal indices; those are reverse-
    translated to user-space `p_id` values here so consumers downstream see
    the original identifiers (with `-1` preserved as the no-link sentinel).
    """
    sorted_orig_p_ids = xnp.asarray(input_data__flat[("p_id",)])[
        input_data__sort_indices
    ]
    return {
        qname: _translate_internal_to_orig_p_ids(
            col=value,
            sorted_orig_p_ids=sorted_orig_p_ids,
            xnp=xnp,
        )
        if _is_endogenous_p_id_pointer(qname)
        else value
        for qname, value in columns.items()
    }


def _is_endogenous_p_id_pointer(qname: str) -> bool:
    leaf = dt.tree_path_from_qname(qname)[-1]
    return leaf.startswith("p_id_")


def _translate_internal_to_orig_p_ids(
    col: IntColumn,
    sorted_orig_p_ids: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Map internal indices back to user-space `p_id`s, collapsing all negatives to -1.

    Any negative value — whether the canonical no-link sentinel `-1` or some
    other negative integer a policy function might emit — collapses to `-1`
    in the output. Treating only `-1` as the sentinel would let other
    negatives silently misroute: `xnp.maximum(col, 0)` would clamp e.g. `-2`
    to `0`, and the gather would return `sorted_orig_p_ids[0]`. Widening the
    guard to `col < 0` keeps that path safe.

    `xnp.maximum(col, 0)` keeps the gathered index in-bounds on JAX (which
    cannot dead-branch on the `xnp.where` predicate); the -1 result is then
    re-injected by the outer `xnp.where`.
    """
    return xnp.where(
        col < 0,
        -1,
        sorted_orig_p_ids[xnp.maximum(col, 0)],
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
) -> QNameResults:
    """The parameters that have been requested as targets.

    Possibly includes outputs of param_functions.
    """
    return {
        pt: specialized_environment__with_processed_params_and_scalars[pt]
        for pt in labels__param_targets
    }
