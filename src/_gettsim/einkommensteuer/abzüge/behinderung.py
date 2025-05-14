"""Tax allowances for the disabled."""

from ttsim import policy_function
from ttsim.config import numpy_or_jax as np


@policy_function(vectorization_strategy="loop")
def pauschbetrag_behinderung_y(
    behinderungsgrad: int, parameter_behindertenpauschbetrag: dict[int, float]
) -> float:
    """Assign tax deduction allowance for handicaped to different handicap degrees."""

    # Get disability degree thresholds
    bins = sorted(parameter_behindertenpauschbetrag)

    # Select corresponding bin.
    selected_bin_index = (
        np.searchsorted(np.asarray([*bins, np.inf]), behinderungsgrad, side="right") - 1
    )
    selected_bin = bins[selected_bin_index]

    # Select appropriate pauschbetrag.
    out = parameter_behindertenpauschbetrag[selected_bin]

    return out
