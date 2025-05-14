"""Tax allowances for the disabled."""

from ttsim import policy_function
from ttsim.config import numpy_or_jax as np


@policy_function(vectorization_strategy="loop")
def pauschbetrag_behinderung_y(
    behinderungsgrad: int, behindertenpauschbetrag: dict[int, float]
) -> float:
    """Assign tax deduction allowance for handicaped to different handicap degrees.

    Parameters
    ----------
    behinderungsgrad
        See basic input variable :ref:`behinderungsgrad <behinderungsgrad>`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """

    # Get disability degree thresholds
    bins = sorted(behindertenpauschbetrag)

    # Select corresponding bin.
    selected_bin_index = (
        np.searchsorted(np.asarray([*bins, np.inf]), behinderungsgrad, side="right") - 1
    )
    selected_bin = bins[selected_bin_index]

    # Select appropriate pauschbetrag.
    out = behindertenpauschbetrag[selected_bin]

    return out
