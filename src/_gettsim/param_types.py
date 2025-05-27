from dataclasses import dataclass

from ttsim.param_objects import ConsecutiveIntLookupTableParamValue


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
class WohngeldBasisformelParamValues:
    skalierungsfaktor: float
    a: ConsecutiveIntLookupTableParamValue
    b: ConsecutiveIntLookupTableParamValue
    c: ConsecutiveIntLookupTableParamValue
    zusatzbetrag_nach_haushaltsgröße: ConsecutiveIntLookupTableParamValue
