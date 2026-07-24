"""Warning classes for ttsim."""


class PotentialCurrencyMismatchWarning(UserWarning):
    """Warn that the user may be passing data in the wrong currency."""


__all__ = [
    "PotentialCurrencyMismatchWarning",
]
