from __future__ import annotations

from pathlib import Path

from ttsim.tt import register_currency, register_unit_builder_levels

ROOT_PATH = Path(__file__).parent

# Middle Earth's currencies. Registered on import so that the [currency]
# dimension has concrete currencies before the policy environment is assembled
# (GEP 10). The castar — Gondor's coin — is the realm's unit of account since
# the currency reform of 2020 and hence the base currency; the silver penny —
# the Shire's coin and the unit of account before the reform — is worth a
# quarter-castar.
register_currency("CASTAR", base=True)
register_currency("SILVER_PENNY", definition="CASTAR / 4")

# Middle Earth's grouping levels. Registered on import so the fluent unit
# builder offers `Unit.X.PER_FAM` / `per_kin` / `per_sp` before the policy
# modules (whose decorators use them) are loaded (GEP 10 compositional units).
register_unit_builder_levels(["sp", "fam", "kin"])

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

__all__ = ["ROOT_PATH"]
