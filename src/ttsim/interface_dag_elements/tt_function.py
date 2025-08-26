from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dags import concatenate_functions

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from ttsim.typing import (
        OrderedQNames,
        QNameData,
        SpecEnvWithPartialledParamsAndScalars,
    )


@interface_input(in_top_level_namespace=True)
def tt_function_set_annotations() -> bool:
    """Whether to set annotations on the tax-transfer function.

    Defaults to true, turn off in case you run into trouble with type annotations when
    modifying the policy environment.
    """


@interface_function(in_top_level_namespace=True)
def tt_function(
    specialized_environment__tt_dag: nx.DiGraph,
    specialized_environment__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    labels__column_targets: OrderedQNames,
    tt_function_set_annotations: bool,
    backend: Literal["numpy", "jax"],
) -> Callable[[QNameData], QNameData]:
    """Return the function calculating the taxes and transfers."""
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
