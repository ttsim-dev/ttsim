from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, ParamSpec, get_args

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

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
        if not isinstance(self.base, (int, float)):
            msg = f"base needs to be a number, got {self.base!r}"
            raise TypeError(msg)
        valid_directions = get_args(ROUNDING_DIRECTION)
        if self.direction not in valid_directions:
            raise ValueError(
                f"`direction` must be one of {valid_directions}, "
                f"got {self.direction!r}",
            )
        if not isinstance(self.to_add_after_rounding, (int, float)):
            msg = f"Additive part must be a number, got {self.to_add_after_rounding!r}"
            raise TypeError(msg)

    def apply_rounding(
        self,
        func: FunctionType[P, FloatColumn],
        xnp: ModuleType,
    ) -> FunctionType[P, FloatColumn]:
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
            else:  # self.direction == "nearest"
                rounded_out = self.base * (xnp.asarray(out) / self.base).round()

            return rounded_out + self.to_add_after_rounding

        return wrapper
