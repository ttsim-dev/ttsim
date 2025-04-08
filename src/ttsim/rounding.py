from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class RoundingDirection(StrEnum):
    """
    Enum for the rounding direction.
    """

    UP = "up"
    DOWN = "down"
    NEAREST = "nearest"


@dataclass
class RoundingSpec:
    base: int | float
    direction: RoundingDirection
    to_add_after_rounding: int | float = 0

    def __post_init__(self):
        """Validate the types of base and to_add_after_rounding."""
        if type(self.base) not in [int, float]:
            raise ValueError(f"base needs to be a number, got {self.base!r}")
        if type(self.to_add_after_rounding) not in [int, float]:
            raise ValueError(
                f"Additive part must be a number, got {self.to_add_after_rounding!r}"
            )

    def apply_rounding(self, func: callable) -> callable:
        """Decorator to round the output of a function.

        Parameters
        ----------
        func
            Function to be rounded.
        name
            Name of the function to be rounded.

        Returns
        -------
        Function with rounding applied.
        """

        # Make sure that signature is preserved.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)

            if self.direction == RoundingDirection.UP:
                rounded_out = self.base * np.ceil(out / self.base)
            elif self.direction == RoundingDirection.DOWN:
                rounded_out = self.base * np.floor(out / self.base)
            elif self.direction == RoundingDirection.NEAREST:
                rounded_out = self.base * (out / self.base).round()

            rounded_out += self.to_add_after_rounding
            return rounded_out

        return wrapper
