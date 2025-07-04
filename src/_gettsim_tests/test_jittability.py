from __future__ import annotations

import contextlib
import datetime
import inspect
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import pytest
from dags import get_free_arguments

from ttsim import Output, main
from ttsim.tt_dag_elements.column_objects_param_function import ColumnFunction

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        SpecEnvWithPartialledParamsAndScalars,
    )

GETTSIM_ROOT = Path(__file__).parent.parent / "_gettsim"


def get_orig_gettsim_objects() -> dict[
    str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
]:
    out = main(
        orig_policy_objects={"root": GETTSIM_ROOT},
        output=Output.names(
            [
                "orig_policy_objects__column_objects_and_param_functions",
                "orig_policy_objects__param_specs",
            ]
        ),
    )
    return {k.replace("orig_policy_objects__", ""): v for k, v in out.items()}


def get_orig_gettsim_column_functions() -> list[ColumnFunction]:
    orig = get_orig_gettsim_objects()["column_objects_and_param_functions"]
    return [(tp, cf) for tp, cf in orig.items() if isinstance(cf, ColumnFunction)]


@lru_cache(maxsize=100)
def cached_specialized_environment(
    date: datetime.date,
    root: Path,
    backend: Literal["numpy", "jax"],
) -> SpecEnvWithPartialledParamsAndScalars:
    return main(
        date=date,
        orig_policy_objects={"root": root},
        backend=backend,
        fail_and_warn=False,
        output=Output.name(
            ("specialized_environment", "with_partialled_params_and_scalars")
        ),
    )


@pytest.mark.skipif_numpy
@pytest.mark.parametrize(
    "tree_path, fun",
    get_orig_gettsim_column_functions(),
    ids=[str(x[0]) for x in get_orig_gettsim_column_functions()],
)
def test_jittable(tree_path, fun, backend, xnp):
    today = datetime.date.today()  # noqa: DTZ011
    date = min(fun.end_date, today)
    qname = dt.qname_from_tree_path(tree_path[:-2] + (fun.leaf_name,))
    env = {qname: cached_specialized_environment(date, GETTSIM_ROOT, backend)[qname]}

    processed_data = {}
    for arg_name in get_free_arguments(env[qname]):
        arg = inspect.signature(env[qname]).parameters[arg_name]
        if "FloatColumn" in arg.annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=float)
        elif "IntColumn" in arg.annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=int)
        elif "BoolColumn" in arg.annotation:
            processed_data[arg_name] = xnp.zeros(1, dtype=bool)
        else:
            raise ValueError(f"Unknown column type: {arg.annotation}")

    with contextlib.suppress(NotImplementedError):
        main(
            date=date,
            specialized_environment={"with_partialled_params_and_scalars": env},
            processed_data=processed_data,
            targets={"qname": [qname]},
            backend=backend,
            output=Output.name(("raw_results", "columns")),
            fail_and_warn=False,
        )
