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
#: the magnitude, below which a quotient counts as that whole base. A product or
#: quotient of two decimal operands lands within about one ulp of its exact value;
#: anything much looser swallows genuine fractions in float32 (16 ulps of 12 290 are
#: 0.023, and 0.204833 * 60000 = 12289.98 would round up to 12290).
_SNAP_TOLERANCE_IN_ULPS = 4

#: Upper bound on that distance in base units. In float32 a few ulps of a large
#: quotient are a sizeable part of a base unit (16 ulps of 181 417 are 0.35), so
#: without the bound a quotient a third below a whole base would be snapped up to it.
_SNAP_TOLERANCE_MAX_IN_BASE_UNITS = 0.1


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
            # Snap near-whole quotients so representation error cannot flip a
            # directed rounding (0.0656 * 40000 is 2623.9999999999995 in binary).
            scaled = xnp.asarray(func(*args, **kwargs)) / self.base
            nearest = xnp.round(scaled)
            tolerance = _SNAP_TOLERANCE_IN_ULPS * xnp.finfo(scaled.dtype).eps
            tolerance = xnp.minimum(
                tolerance * xnp.maximum(xnp.abs(scaled), 1.0),
                _SNAP_TOLERANCE_MAX_IN_BASE_UNITS,
            )
            scaled = xnp.where(xnp.abs(scaled - nearest) <= tolerance, nearest, scaled)
            round_func = {"up": xnp.ceil, "down": xnp.floor, "nearest": xnp.round}
            rounded_out = self.base * round_func[self.direction](scaled)
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
