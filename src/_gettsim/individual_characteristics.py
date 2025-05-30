from __future__ import annotations

import datetime

from ttsim import policy_function
from ttsim.config import numpy_or_jax as np


@policy_function()
def geburtsdatum(
    geburtsjahr: int,
    geburtsmonat: int,
    geburtstag: int,
) -> np.datetime64:
    """Create date of birth datetime variable."""
    return np.datetime64(
        datetime.datetime(
            geburtsjahr,
            geburtsmonat,
            geburtstag,
        )
    ).astype("datetime64[D]")


@policy_function()
def alter_bis_24(alter: int) -> bool:
    """Age is 24 years at most.

    Trivial, but necessary in order to use the target for aggregation.
    """
    return alter <= 24
