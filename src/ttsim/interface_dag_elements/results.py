from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dags.tree as dt
import numpy
import pandas as pd

from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.typing import (
        FlatData,
        IntColumn,
        NestedData,
        NestedStrings,
        QNameData,
    )


def _restore_original_row_order_in_nested_data(
    nested_data: NestedData, input_data__sort_indices: IntColumn
) -> NestedData:
    """Restore the original row order in nested data structures.

    Args:
        nested_data: The nested data with rows in sorted order.
        input_data__sort_indices: Sort indices for restoring original row order.

    Returns:
        Nested data with rows restored to their original order.
    """
    num_rows = len(input_data__sort_indices)

    # Create inverse permutation: if sort_indices[i] = j, then restore_order[j] = i
    restore_order = numpy.empty(len(input_data__sort_indices), dtype=int)
    restore_order[input_data__sort_indices] = numpy.arange(
        len(input_data__sort_indices)
    )

    def _restore_nested_order(data: NestedData) -> NestedData:
        """Recursively restore order in nested structures."""
        # Check if this is a nested mapping (has .items() method)
        if hasattr(data, "items"):
            return cast(
                "NestedData", {k: _restore_nested_order(v) for k, v in data.items()}
            )

        # Restore original order if they match row count
        if (
            hasattr(data, "__array__")
            and hasattr(data, "shape")
            and len(data.shape) >= 1
            and data.shape[0] == num_rows
        ):
            return cast("NestedData", data[restore_order])

        # For anything else (scalars, etc.), return as-is
        return data

    return _restore_nested_order(nested_data)


@interface_function()
def tree(
    raw_results__columns: QNameData,
    raw_results__params: QNameData,
    raw_results__from_input_data: QNameData,
    input_data__flat: FlatData,
    input_data__sort_indices: IntColumn,
) -> NestedData:
    """The combined results as a tree with original row order restored.

    The transformed id's are converted back to their original values, and the
    row order is restored to match the original input data order.

    """
    # Combine the three result sources
    raw_results_combined = {
        **raw_results__columns,
        **raw_results__params,
        **raw_results__from_input_data,
    }

    out = {}
    for k, v in raw_results_combined.items():
        path = dt.tree_path_from_qname(k)
        if path in input_data__flat and (
            path[-1].endswith("_id") or path[-1].startswith("p_id_")
        ):
            out[k] = input_data__flat[path]
        else:
            out[k] = v

    # Convert to tree structure first
    tree_data = dt.unflatten_from_qnames(out)

    # Then restore original row order
    return _restore_original_row_order_in_nested_data(
        tree_data, input_data__sort_indices
    )


@interface_function()
def df_with_mapper(
    tree: NestedData,
    input_data__flat: FlatData,
    tt_targets__tree: NestedStrings,
) -> pd.DataFrame:
    """The results DataFrame with mapped column names.

    Args:
        tree:
            The results of a TTSIM run with original row order already restored.
        input_data__flat:
            The input data containing original p_ids.
        tt_targets__tree:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns
    -------
        A DataFrame.
    """
    # Use original p_ids directly since tree already has the correct row order
    original_p_ids = input_data__flat[("p_id",)]
    original_dtype = original_p_ids.dtype

    # Convert to numpy array with original dtype to ensure
    # consistent pandas Index dtype
    original_p_ids_array = numpy.asarray(original_p_ids, dtype=original_dtype)

    # Create a temporary data structure with original p_ids for the
    # DataFrame index (tree data is already in correct order)
    data_with_original_p_id = {"p_id": original_p_ids_array}

    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=tree,
        nested_outputs_df_column_names=tt_targets__tree,
        data_with_p_id=data_with_original_p_id,
    )


@interface_function()
def df_with_nested_columns(
    tree: NestedData,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """The results DataFrame with nested column names corresponding to tree paths.

    Args:
        tree:
            The results of a TTSIM run with original row order already restored.
        input_data__flat:
            The flat input data containing original p_ids.

    Returns
    -------
    A DataFrame with a hierarchical index in the column dimension.
    """
    # Use original p_ids directly since tree already has the correct row order
    original_p_ids = input_data__flat[("p_id",)]
    original_dtype = original_p_ids.dtype

    # Convert to numpy array with original dtype if available to ensure
    # consistent pandas Index dtype
    if original_dtype is not None:
        original_p_ids_array = numpy.asarray(original_p_ids, dtype=original_dtype)
    else:
        original_p_ids_array = numpy.asarray(original_p_ids)

    return nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        index=pd.Index(original_p_ids_array, name="p_id"),
    )
