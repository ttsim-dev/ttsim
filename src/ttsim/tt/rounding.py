import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Literal, ParamSpec, get_args

from beartype import beartype

from ttsim._beartype_conf import ROUNDING_SPEC_CONF
from ttsim.tt.type_resolution import build_beartype_checkable_wrapper
from ttsim.tt.units import (
    CompositeUnit,
    currency_conversion_factor,
    token_source_currency,
)
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


@beartype(conf=ROUNDING_SPEC_CONF)
@dataclass(frozen=True)
class RoundingSpec:
    base: int | float
    direction: ROUNDING_DIRECTION
    to_add_after_rounding: int | float = 0
    reference: str | None = None
    unit: CompositeUnit | None = None
    """The fully-spelled unit ``base`` and ``to_add_after_rounding`` are written
    in. Mandatory for a spec on a currency-valued function — the magnitudes are
    statutory numbers written in a concrete currency, exactly like a parameter's
    (``Unit.DM.PER_YEAR``) — and its composite must equal the function's declared
    unit with the agnostic base swapped for the concrete currency. Stays ``None``
    on a non-currency function: there is nothing to convert (GEP 10)."""

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

    def in_run_currency(self, run_currency: str | None) -> "RoundingSpec":
        """This spec with its magnitudes restated in the run currency (GEP 10).

        Keeps the rounding step statutorily exact under a currency changeover:
        rounding down to multiples of 54 DM in a EUR run becomes rounding down
        to multiples of ``54 / 1.95583`` EUR. Returns ``self`` when there is
        nothing to convert — no run currency, no declared unit, the run
        currency itself, or a declaration that does not pin down a registered
        currency (the unit checks reject those; the conversion never guesses).
        """
        if run_currency is None or self.unit is None:
            return self
        source = token_source_currency(self.unit)
        if source is None or source == run_currency:
            return self
        factor = currency_conversion_factor(
            source_currency=source, run_currency=run_currency
        )
        return replace(
            self,
            base=self.base * factor,
            to_add_after_rounding=self.to_add_after_rounding * factor,
            unit=replace(self.unit, base=run_currency.upper()),
        )

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
            out = func(*args, **kwargs)

            if self.direction == "up":
                rounded_out = self.base * xnp.ceil(out / self.base)
            elif self.direction == "down":
                rounded_out = self.base * xnp.floor(out / self.base)
            else:  # self.direction == "nearest"
                rounded_out = self.base * (xnp.asarray(out) / self.base).round()

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
