"""Input columns."""

from ttsim import policy_input


@policy_input()
def bezieht_rente() -> bool:
    """Draws public pension benefits."""
    return False


@policy_input()
def entgeltpunkte_ost() -> float:
    """Earnings points for pension claim accumulated in Eastern states."""
    return 0


@policy_input()
def entgeltpunkte_west() -> float:
    """Earnings points for pension claim accumulated in Western states."""
    return 0


@policy_input()
def ersatzzeiten_monate() -> float:
    """Total months during military, persecution/escape, internment, and consecutive sickness."""
    return 0


@policy_input()
def freiwillige_beitragsmonate() -> float:
    """Total months of voluntary pensioninsurance contributions."""
    return 0


@policy_input()
def jahr_renteneintritt() -> int:
    """Year of pension claiming."""
    return 0


@policy_input()
def monat_renteneintritt() -> int:
    """Month of retirement."""
    return 0


@policy_input()
def kinderberücksichtigungszeiten_monate() -> float:
    """Total months of childcare till age 10."""
    return 0


@policy_input()
def krankheitszeiten_ab_16_bis_24_monate() -> float:
    """Total months of sickness between age 16 and 24."""
    return 0


@policy_input()
def monate_geringfügiger_beschäftigung() -> float:
    """Total months of marginal employment (w/o mandatory contributions)."""
    return 0


@policy_input()
def monate_in_arbeitslosigkeit() -> float:
    """Total months of unemployment (registered)."""
    return 0


@policy_input()
def monate_in_arbeitsunfähigkeit() -> float:
    """Total months of sickness, rehabilitation, measures for worklife participation(Teilhabe)."""
    return 0


@policy_input()
def monate_in_ausbildungssuche() -> float:
    """Total months of apprenticeship search."""
    return 0


@policy_input()
def monate_in_mutterschutz() -> float:
    """Total months of maternal protections."""
    return 0


@policy_input()
def monate_in_schulausbildung() -> float:
    """Months of schooling (incl college, unifrom age 17, max. 8 years)."""
    return 0


@policy_input()
def monate_mit_bezug_entgeltersatzleistungen_wegen_arbeitslosigkeit() -> float:
    """Total months of unemployment (only time of Entgeltersatzleistungen, not ALGII),i.e. Arbeitslosengeld, Unterhaltsgeld, Übergangsgeld."""
    return 0


@policy_input()
def pflichtbeitragsmonate() -> float:
    """Total months of mandatory pension insurance contributions."""
    return 0


@policy_input()
def private_rente_betrag_m() -> float:
    """Amount of monthly private pension."""
    return 0


@policy_input()
def pflegeberücksichtigungszeiten_monate() -> float:
    """Total months of home care provision (01.01.1992-31.03.1995)."""
    return 0
