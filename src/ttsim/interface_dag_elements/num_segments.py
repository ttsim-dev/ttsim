from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.typing import QNameData


@interface_function(in_top_level_namespace=True)
def num_segments(processed_data: QNameData) -> int:
    """The number of segments for segment sums in jax."""

    if processed_data:
        # After processing the data, we know that the number of ids is at most the
        # length of the data.
        return len(next(iter(processed_data.values())))
    # Leave at a recognisable value; just used in jittability tests.
    return 11111
