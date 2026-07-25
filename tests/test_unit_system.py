"""Representative unit system owned by the TTSIM test suite."""

from ttsim.tt.units import UnitSystem, register_unit_builder_levels

register_unit_builder_levels(("bg", "fam", "hh", "kin"))

TEST_UNIT_SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"},
)
