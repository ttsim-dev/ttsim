from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedInputStructureDict,
        OrderedQNames,
        QNameSpecializedEnvironment0,
        UnorderedQNames,
    )


@interface_function()
def input_data_dtypes(
    specialized_environment__with_partialled_params_and_scalars: QNameSpecializedEnvironment0,  # noqa: E501
    targets__qname: OrderedQNames,
    labels__top_level_namespace: UnorderedQNames,
) -> NestedInputStructureDict:
    return dt.create_tree_with_input_types(
        functions=dt.unflatten_from_qnames(
            specialized_environment__with_partialled_params_and_scalars,
        ),
        targets=targets__qname,
        top_level_inputs=labels__top_level_namespace,
    )
