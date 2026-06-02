from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.typing import QNameData


@interface_function(in_top_level_namespace=True)
def len_p_id(processed_data: QNameData) -> int:
    """The length of `p_id`, i.e. the number of observations (``n_obs``).

    Every column in `processed_data` has this length, so it is read off the first one.
    It serves two purposes:

    - It is the segment count for jax segment sums. The number of distinct groups for
      any group identifier is at most ``n_obs``, so ``n_obs`` is a safe upper bound;
      aggregation functions receive it under the name ``num_segments`` (see
      `specialized_environment.with_partialled_params_and_scalars`).
    - It is the population length used to broadcast scalar inputs to non-vectorized
      functions (see `specialized_environment._broadcast_scalar_columns_at_call_time`).
    """
    if processed_data:
        return len(next(iter(processed_data.values())))
    # Leave at a recognisable value; just used in jittability tests.
    return 11111
