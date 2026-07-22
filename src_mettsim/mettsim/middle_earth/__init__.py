from __future__ import annotations

from pathlib import Path

from ttsim.tt import UnitSystem

ROOT_PATH = Path(__file__).parent

# Middle Earth's unit system, built on import so the [currency] dimension has
# concrete currencies and the fluent builder offers `TTSIMUnit.X.PER_FAM` / `per_kin`
# / `per_sp` before the policy modules (whose decorators use them) are loaded
# (GEP 10). The castar — Gondor's coin — is the realm's unit of account since
# the currency reform of 2020 and hence the base currency; the silver penny —
# the Shire's coin and the unit of account before the reform — is worth a
# quarter-castar. The statutory-currency mapping encodes the reform: statutes
# denominate their numbers in silver pennies until the end of 2019 and in
# castar from 2020 on, and each era's computations run in its own currency.
UNIT_SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"},
    grouping_levels=["sp", "fam", "kin"],
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
