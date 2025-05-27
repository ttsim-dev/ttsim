"""Social insurance contributions."""

from __future__ import annotations

from ttsim import policy_function


@policy_function()
def beiträge_versicherter_m(
    sozialversicherung__pflege__beitrag__betrag_versicherter_m: float,
    sozialversicherung__kranken__beitrag__betrag_versicherter_m: float,
    sozialversicherung__rente__beitrag__betrag_versicherter_m: float,
    sozialversicherung__arbeitslosen__beitrag__betrag_versicherter_m: float,
) -> float:
    """Sum of social insurance contributions paid by the insured person."""
    return (
        sozialversicherung__pflege__beitrag__betrag_versicherter_m
        + sozialversicherung__kranken__beitrag__betrag_versicherter_m
        + sozialversicherung__rente__beitrag__betrag_versicherter_m
        + sozialversicherung__arbeitslosen__beitrag__betrag_versicherter_m
    )


@policy_function()
def beiträge_arbeitgeber_m(
    sozialversicherung__pflege__beitrag__betrag_arbeitgeber_m: float,
    sozialversicherung__kranken__beitrag__betrag_arbeitgeber_m: float,
    sozialversicherung__rente__beitrag__betrag_arbeitgeber_m: float,
    sozialversicherung__arbeitslosen__beitrag__betrag_arbeitgeber_m: float,
) -> float:
    """Sum of employer's social insurance contributions."""
    return (
        sozialversicherung__pflege__beitrag__betrag_arbeitgeber_m
        + sozialversicherung__kranken__beitrag__betrag_arbeitgeber_m
        + sozialversicherung__rente__beitrag__betrag_arbeitgeber_m
        + sozialversicherung__arbeitslosen__beitrag__betrag_arbeitgeber_m
    )


@policy_function()
def beiträge_gesamt_m(
    beiträge_versicherter_m: float,
    beiträge_arbeitgeber_m: float,
) -> float:
    """Sum of employer's and insured person's social insurance contributions."""
    return beiträge_versicherter_m + beiträge_arbeitgeber_m
