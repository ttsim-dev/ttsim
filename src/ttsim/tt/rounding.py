from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, ParamSpec, get_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ttsim.typing import FloatColumn


ROUNDING_DIRECTION = Literal["up", "down", "nearest"]

P = ParamSpec("P")


@dataclass
class RoundingSpec:
    base: int | float
    direction: ROUNDING_DIRECTION
    to_add_after_rounding: int | float = 0
    reference: str | None = None

    def __post_init__(self) -> None:
        """Validate the types of base and to_add_after_rounding."""
        if type(self.base) not in [int, float]:
            raise ValueError(f"base needs to be a number, got {self.base!r}")
        valid_directions = get_args(ROUNDING_DIRECTION)
        if self.direction not in valid_directions:
            raise ValueError(
                f"`direction` must be one of {valid_directions}, "
                f"got {self.direction!r}",
            )
        if type(self.to_add_after_rounding) not in [int, float]:
            raise ValueError(
                f"Additive part must be a number, got {self.to_add_after_rounding!r}",
            )

    def apply_rounding(
        self,
        func: Callable[P, FloatColumn],
        xnp: ModuleType,
    ) -> Callable[P, FloatColumn]:
        """Decorator to round the output of a function.

        Parameters
        ----------
        func
            Function to be rounded.
        xnp
            The computing module to use.

        Returns
        -------
        Function with rounding applied.
        """

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FloatColumn:
            out = func(*args, **kwargs)

            if self.direction == "up":
                rounded_out = self.base * xnp.ceil(out / self.base)
            elif self.direction == "down":
                rounded_out = self.base * xnp.floor(out / self.base)
            elif self.direction == "nearest":
                rounded_out = self.base * (xnp.asarray(out) / self.base).round()

            return rounded_out + self.to_add_after_rounding

        return wrapper
