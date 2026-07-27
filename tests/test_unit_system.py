"""Representative unit system owned by the TTSIM test suite."""

from ttsim.tt.units import Currency, UnitSystem, register_unit_builder_levels

register_unit_builder_levels(("bg", "fam", "hh", "kin"))

TEST_UNIT_SYSTEM = UnitSystem(
    currencies={
        "CASTAR": Currency(statutory_from="2020-01-01"),
        "SILVER_PENNY": Currency(value="CASTAR / 4", statutory_from="0001-01-01"),
    },
)
