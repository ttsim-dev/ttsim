"""Middle-Earth Taxes and Transfers Simulator.

TTSIM specification for testing purposes. Taxes and transfer names follow a law-to-code
approach based on the Gondorian tax code.
"""

from pathlib import Path

RESOURCE_DIR = Path(__file__).parent

SUPPORTED_GROUPINGS = ("fam", "sp", "hh")
