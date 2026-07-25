from __future__ import annotations

from pathlib import Path

from ttsim.tt import UnitSystem, register_unit_builder_levels

ROOT_PATH = Path(__file__).parent

# Register the unit-builder levels for Middle Earth. The levels are:
register_unit_builder_levels(["sp", "fam", "kin"])

# Middle Earth's unit system, built on import so the [currency] dimension has
# concrete currencies (GEP 10). The castar — Gondor's coin — is the realm's
# unit of account since the currency reform of 2020 and hence the base
# currency; the silver penny — the Shire's coin and the unit of account before
# the reform — is worth a quarter-castar.
UNIT_SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"},
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
