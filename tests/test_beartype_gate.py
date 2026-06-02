"""The `TTSIM_BEARTYPE_CLAW` switch must turn *all* beartype checking on/off.

GEP 9 makes runtime type checking on by default; `TTSIM_BEARTYPE_CLAW=0` is the
opt-out and must restore pre-GEP behaviour — no package claw *and* no perimeter
checking. A single flag (`ttsim._beartype_conf.RUNTIME_TYPE_CHECKING_ENABLED`) drives
both the package claw and the strategy of every `@beartype(conf=...)` conf, so
the off position reduces all of them to beartype's no-op `O0` strategy.

The gate and the confs are evaluated at import time, so each switch position
runs in a fresh subprocess.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Imports ttsim under the inherited env, then reports the gate flag, the
# package-claw and a perimeter conf's strategy, and whether the synthesized
# forwarder actually rejects a mistyped argument (the live-behaviour check).
_PROBE = textwrap.dedent(
    """
    import ttsim
    from ttsim._beartype_conf import (
        ENTRY_POINT_CONF,
        INTERNAL_CONF,
        RUNTIME_TYPE_CHECKING_ENABLED,
    )
    from ttsim.tt.type_resolution import build_beartype_checkable_wrapper
    from beartype.roar import BeartypeCallHintViolation

    wrapper = build_beartype_checkable_wrapper(
        wrapped=(lambda x: x),
        annotations={"x": "IntColumn", "return": "IntColumn"},
        node_name="probe",
    )
    try:
        wrapper("not-a-column")
        forwarder_checks = 0
    except BeartypeCallHintViolation:
        forwarder_checks = 1

    print(
        f"enabled={int(RUNTIME_TYPE_CHECKING_ENABLED)};"
        f"internal={INTERNAL_CONF.strategy.name};"
        f"perimeter={ENTRY_POINT_CONF.strategy.name};"
        f"forwarder_checks={forwarder_checks}"
    )
    """
)


def _probe(claw: str | None) -> dict[str, str]:
    """Import ttsim in a fresh process under the given switch value.

    Args:
        claw: Value for `TTSIM_BEARTYPE_CLAW`, or `None` to leave it unset.

    Returns:
        The parsed `key=value` state the child process reports.
    """
    env = {
        key: value for key, value in os.environ.items() if key != "TTSIM_BEARTYPE_CLAW"
    }
    if claw is not None:
        env["TTSIM_BEARTYPE_CLAW"] = claw
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(item.split("=") for item in proc.stdout.strip().split(";"))


@pytest.mark.parametrize("claw", [None, "1"])
def test_runtime_checking_on_by_default(claw: str | None) -> None:
    """Unset (the default) or `=1` keeps the claw and perimeter checks live."""
    assert _probe(claw) == {
        "enabled": "1",
        "internal": "On",
        "perimeter": "On",
        "forwarder_checks": "1",
    }


def test_runtime_checking_fully_off_when_opted_out() -> None:
    """`TTSIM_BEARTYPE_CLAW=0` disables every check (pre-GEP behaviour)."""
    assert _probe("0") == {
        "enabled": "0",
        "internal": "O0",
        "perimeter": "O0",
        "forwarder_checks": "0",
    }
