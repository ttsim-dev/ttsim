from __future__ import annotations

from pathlib import Path

from ttsim.tt import Currency, UnitSystem, register_unit_builder_levels

ROOT_PATH = Path(__file__).parent

register_unit_builder_levels(["fam", "kin"])

UNIT_SYSTEM = UnitSystem(
    currencies={
        "CASTAR": Currency(statutory_from="2020-01-01"),
        "SILVER_PENNY": Currency(value="CASTAR / 4", statutory_from="0001-01-01"),
    },
)

COLORMAP: dict[tuple[str, ...] | str, str] = {
    ("housing_benefits",): "red",
    ("orc_hunting_bounty",): "green",
    ("payroll_tax",): "gold",
    ("payroll_tax", "child_tax_credit"): "orange",
    ("payroll_tax", "income"): "yellow",
    ("wealth_tax",): "blue",
    ("property_tax",): "dodgerblue",
    ("top-level",): "navy",
}

__all__ = ["ROOT_PATH", "UNIT_SYSTEM"]
