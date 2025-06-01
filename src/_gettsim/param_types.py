from dataclasses import dataclass
from typing import Literal

from ttsim.param_objects import ConsecutiveInt1dLookupTableParamValue


@dataclass(frozen=True)
class Altersgrenzen:
    min_alter: int
    max_alter: int


@dataclass(frozen=True)
class SatzMitAltersgrenzen:
    satz: float
    altersgrenzen: Altersgrenzen


@dataclass(frozen=True)
class ElementExistenzminimum:
    single: float
    paar: float
    kind: float


@dataclass(frozen=True)
class ElementExistenzminimumNurKind:
    kind: float


@dataclass(frozen=True)
class ExistenzminimumNachAufwendungenOhneBildungUndTeilhabe:
    regelsatz: ElementExistenzminimum
    kosten_der_unterkunft: ElementExistenzminimum
    heizkosten: ElementExistenzminimum


@dataclass(frozen=True)
class ExistenzminimumNachAufwendungenMitBildungUndTeilhabe:
    regelsatz: ElementExistenzminimum
    kosten_der_unterkunft: ElementExistenzminimum
    heizkosten: ElementExistenzminimum
    bildung_und_teilhabe: ElementExistenzminimumNurKind


@dataclass(frozen=True)
class SGBIIRegelsatzAnteilErwachsen:
    je_erwachsener_bei_zwei_erwachsenen: float
    je_erwachsener_ab_drei_erwachsene: float


@dataclass(frozen=True)
class SGBIIRegelsatzAnteilKind:
    anteil: float
    min_alter: int
    max_alter: int


@dataclass(frozen=True)
class SGBIIRegelsatzAnteilKindNachAlter:
    kleinkind: SGBIIRegelsatzAnteilKind
    schulkind: SGBIIRegelsatzAnteilKind
    jugendliche_und_junge_erwachsene: SGBIIRegelsatzAnteilKind


@dataclass(frozen=True)
class SGBIIRegelsatzAnteilsbasiert:
    basissatz: float
    erwachsen: SGBIIRegelsatzAnteilErwachsen
    kind: SGBIIRegelsatzAnteilKindNachAlter


@dataclass(frozen=True)
class SGBIIRegelsatzNachRegelbedarfsstufen:
    """Regelsatz as a fraction of the Basissatz."""

    rbs_1: float
    rbs_2: float
    rbs_3: float
    rbs_4: SatzMitAltersgrenzen
    rbs_5: SatzMitAltersgrenzen
    rbs_6: SatzMitAltersgrenzen


@dataclass(frozen=True)
class BerechtigteWohnflächeEigentum:
    anzahl_personen_zu_fläche: dict[int, float]
    je_weitere_person: float
    max_anzahl_direkt: int


ErziehungsgeldSätze = Literal["regelsatz", "budgetsatz"]


@dataclass(frozen=True)
class EinkommensgrenzeErziehungsgeld:
    regulär_alleinerziehend: dict[ErziehungsgeldSätze, float]
    regulär_paar: dict[ErziehungsgeldSätze, float]
    reduziert_alleinerziehend: dict[ErziehungsgeldSätze, float]
    reduziert_paar: dict[ErziehungsgeldSätze, float]


@dataclass(frozen=True)
class WohngeldBasisformelParamValues:
    skalierungsfaktor: float
    a: ConsecutiveInt1dLookupTableParamValue
    b: ConsecutiveInt1dLookupTableParamValue
    c: ConsecutiveInt1dLookupTableParamValue
    zusatzbetrag_nach_haushaltsgröße: ConsecutiveInt1dLookupTableParamValue
