"""Middle-Earth Taxes and Transfers Simulator.

TTSIM specification for testing purposes. Taxes and transfer names follow a law-to-code
approach based on the Gondorian tax code.
"""

from pathlib import Path

METTSIM_RESSOURCE_DIR = Path(__file__).parent


FOREIGN_KEYS = (
    ("payroll_tax", "p_id_spouse"),
    ("p_id_parent_1",),
    ("p_id_parent_2",),
)
