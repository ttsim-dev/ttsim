"""Input columns."""

from ttsim import policy_input


@policy_input()
def anwartschaftszeit() -> bool:
    """At least 12 months of unemployment contributions in the 30 months before claiming unemployment insurance."""
    return False


@policy_input()
def arbeitssuchend() -> bool:
    """Looking for employment."""
    return False


@policy_input()
def monate_durchgängigen_bezugs_von_arbeitslosengeld() -> float:
    """Number of months the individual already receives Arbeitslosengeld without interruption."""
    return 0


@policy_input()
def monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren() -> float:
    """Months of subjection to compulsory insurance in the 5 years before claiming unemployment."""
    return 0
