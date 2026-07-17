from __future__ import annotations

from ttsim.interface_dag_elements.interface_node_objects import interface_input
from ttsim.tt.currencies import UnitSystem


@interface_input(in_top_level_namespace=True)
def unit_system() -> UnitSystem:
    """The policy system's currencies, statutory-currency mapping, and levels.

    A user-supplied leaf, like ``orig_policy_objects__root``: a policy package
    declares one :class:`~ttsim.tt.currencies.UnitSystem` at import and passes it
    to ``main(unit_system=...)``. Packages wrapping ``main`` (GETTSIM) default it
    to their own, so their users never name it.
    """
