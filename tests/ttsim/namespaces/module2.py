"""Test namespace."""

from ttsim.function_types import policy_function


@policy_function()
def f(g: int, module2_params: dict[str, int]) -> int:  # noqa: ARG001
    return module2_params["a"] + module2_params["b"]


@policy_function()
def g(module1__f: int, module2_params: dict[str, int]) -> int:
    return module1__f + module2_params["c"]


FUNCTIONS = {
    "module2": {
        "f": f,
        "g": g,
    }
}
