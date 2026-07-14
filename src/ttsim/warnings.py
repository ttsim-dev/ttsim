"""Warning classes for ttsim.

Warnings live here rather than in the interface-DAG modules that issue them:
the interface-DAG loader executes those modules under bare short names, so a
class defined there would not survive pickling across processes (e.g.
pytest-xdist workers).
"""


class PotentialCurrencyMismatchWarning(UserWarning):
    """Warn that the user may be passing data in the wrong currency."""


__all__ = [
    "PotentialCurrencyMismatchWarning",
]
