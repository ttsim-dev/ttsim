from __future__ import annotations

from dags import concatenate_functions
from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttsim.typing import (
        SpecEnvWithPartialledParamsAndScalars,
        QNameData,
        OrderedQNames,
        Literal,
    )
    from collections.abc import Callable
    import networkx as nx


@interface_input(in_top_level_namespace=True)
def tt_function_set_annotations() -> bool:
    """Whether to set annotations on the tax-transfer function.

    Defaults to true, turn off in case you run into trouble with type annotations when
    modifying the policy environment.
    """


@interface_function(in_top_level_namespace=True)
def tt_function(
    specialized_environment__tt_dag: nx.DiGraph,
    specialized_environment__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,
    labels__column_targets: OrderedQNames,
    backend: Literal["numpy", "jax"],
    tt_function_set_annotations: bool,
) -> Callable[[QNameData], QNameData]:
    """Returns a function that takes a dictionary of arrays and unpacks them as keyword arguments."""
    ttf_with_keyword_args = concatenate_functions(
        dag=specialized_environment__tt_dag,
        functions=specialized_environment__with_partialled_params_and_scalars,
        targets=list(labels__column_targets),
        return_type="dict",
        aggregator=None,
        enforce_signature=True,
        set_annotations=tt_function_set_annotations,
    )

    if backend == "jax":
        import jax  # noqa: PLC0415

        ttf_with_keyword_args = jax.jit(ttf_with_keyword_args)

    def wrapper(processed_data: QNameData) -> QNameData:
        return ttf_with_keyword_args(**processed_data)

    return wrapper