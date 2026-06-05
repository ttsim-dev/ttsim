from __future__ import annotations

from pathlib import Path

from ttsim.tt import register_currency

ROOT_PATH = Path(__file__).parent

# Middle Earth's currency. Registered on import so that the [currency] dimension
# has a concrete base currency before the policy environment is assembled
# (GEP 10). The castar — Gondor's coin — is the realm's unit of account and
# hence the base currency.
register_currency("castar", base=True)

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
