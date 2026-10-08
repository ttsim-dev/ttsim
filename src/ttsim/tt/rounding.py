import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, ParamSpec, get_args

from beartype import beartype

from ttsim._beartype_conf import ROUNDING_SPEC_CONF
from ttsim.tt.type_resolution import build_beartype_checkable_wrapper
from ttsim.tt.units import CompositeUnit
from ttsim.typing import FloatColumn

ROUNDING_DIRECTION = Literal["up", "down", "nearest"]

P = ParamSpec("P")


# Drop annotations from the inner `*args, **kwargs` rounding wrapper. The
# outer real-parameter forwarder built by `build_beartype_checkable_wrapper`
# carries the synthesised column-typed signature that beartype actually
# checks, so the inner layer stays untyped to avoid double-resolution.
_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS: tuple[str, ...] = tuple(
    a
    for a in functools.WRAPPER_ASSIGNMENTS
    if a not in ("__annotations__", "__annotate__")
)


#: Distance to the nearest whole base, in units of the dtype's machine epsilon times
#: the magnitude, below which a quotient counts as that whole base.
_SNAP_TOLERANCE_IN_ULPS = 16


@beartype(conf=ROUNDING_SPEC_CONF)
@dataclass(frozen=True)
class RoundingSpec:
    base: int | float
    direction: ROUNDING_DIRECTION
    to_add_after_rounding: int | float = 0
    reference: str | None = None
    unit: CompositeUnit | None = None

    def __post_init__(self) -> None:
        """Validate the types of base and to_add_after_rounding."""
        if not isinstance(self.base, (int, float)):
            msg = f"base needs to be a number, got {self.base!r}"
            raise TypeError(msg)
        if self.base <= 0:
            msg = f"base must be positive, got {self.base!r}"
            raise ValueError(msg)
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
        func: Callable[P, FloatColumn],
        xnp: ModuleType,
    ) -> Callable[P, FloatColumn]:
        """Decorator to round the output of a function.

        Args:
            func: Function to be rounded.
            xnp: The computing module to use.

        Returns:
            Function with rounding applied.
        """

        @functools.wraps(func, assigned=_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FloatColumn:
            # A quotient within 16 ulps of a whole number is treated as that
            # whole number, so directed rounding does not react to the representation
            # error of a decimal operand (0.0656 * 40000 is 2623.9999999999995 in
            # binary floating point, and must round down to 2624, not 2623). The
            # tolerance scales with the magnitude and the dtype's precision, so it
            # stays above the error for large amounts and in float32 alike.
            scaled = xnp.asarray(func(*args, **kwargs)) / self.base
            nearest = xnp.round(scaled)
            tolerance = (
                _SNAP_TOLERANCE_IN_ULPS
                * xnp.finfo(scaled.dtype).eps
                * xnp.maximum(xnp.abs(scaled), 1.0)
            )
            scaled = xnp.where(xnp.abs(scaled - nearest) <= tolerance, nearest, scaled)

            if self.direction == "up":
                rounded_out = self.base * xnp.ceil(scaled)
            elif self.direction == "down":
                rounded_out = self.base * xnp.floor(scaled)
            else:  # self.direction == "nearest"
                rounded_out = self.base * xnp.round(scaled)

            return rounded_out + self.to_add_after_rounding

        # Synthesise the typed outer forwarder. Inputs mirror the wrapped
        # function's signature; the return is always `FloatColumn` because
        # rounding only applies to float-valued column functions.
        func_sig = inspect.signature(func)
        annotations: dict[str, object] = {
            name: param.annotation
            for name, param in func_sig.parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
        annotations["return"] = "FloatColumn"
        return build_beartype_checkable_wrapper(
            wrapper,
            annotations=annotations,
            node_name=getattr(func, "__name__", "_rounded_node"),
        )
