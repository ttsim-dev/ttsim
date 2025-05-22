"""Basic child allowance (Kindergeld)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import AggType, agg_by_p_id_function, join, policy_function

if TYPE_CHECKING:
    from ttsim.config import numpy_or_jax as np


@agg_by_p_id_function(agg_type=AggType.SUM)
def anzahl_ansprüche(
    grundsätzlich_anspruchsberechtigt: bool, p_id_empfänger: int, p_id: int
) -> int:
    pass


@policy_function(start_date="2023-01-01", leaf_name="betrag_m")
def betrag_ohne_staffelung_m(
    anzahl_ansprüche: int,
    satz_einheitlich: float,
) -> float:
    """Sum of Kindergeld for eligible children.

    Kindergeld claim is the same for each child, i.e. increases linearly with the number
    of children.

    """

    return satz_einheitlich * anzahl_ansprüche


@policy_function(
    end_date="2022-12-31", leaf_name="betrag_m", vectorization_strategy="loop"
)
def betrag_gestaffelt_m(
    anzahl_ansprüche: int,
    satz_gestaffelt: dict[int, float],
) -> float:
    """Sum of Kindergeld that parents receive for their children.

    Kindergeld claim for each child depends on the number of children Kindergeld is
    being claimed for.

    """

    if anzahl_ansprüche == 0:
        sum_kindergeld = 0.0
    else:
        sum_kindergeld = sum(
            satz_gestaffelt[(min(i, max(satz_gestaffelt)))]
            for i in range(1, anzahl_ansprüche + 1)
        )

    return sum_kindergeld


@policy_function(
    end_date="2011-12-31",
    leaf_name="grundsätzlich_anspruchsberechtigt",
)
def grundsätzlich_anspruchsberechtigt_nach_lohn(
    alter: int,
    in_ausbildung: bool,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    altersgrenze: dict[str, int],
    maximales_einkommen_des_kindes: float,
) -> bool:
    """Determine kindergeld eligibility for an individual child depending on kids wage.

    Until 2011, there was an income ceiling for children
    returns a boolean variable whether a specific person is a child eligible for
    child benefit

    """
    return (alter < altersgrenze["ohne_bedingungen"]) or (
        (alter < altersgrenze["mit_bedingungen"])
        and in_ausbildung
        and (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
            <= maximales_einkommen_des_kindes
        )
    )


@policy_function(
    start_date="2012-01-01",
    leaf_name="grundsätzlich_anspruchsberechtigt",
)
def grundsätzlich_anspruchsberechtigt_nach_stunden(
    alter: int,
    in_ausbildung: bool,
    arbeitsstunden_w: float,
    altersgrenze: dict[str, int],
    maximale_arbeitsstunden_des_kindes: float,
) -> bool:
    """Determine kindergeld eligibility for an individual child depending on working
    hours.

    The current eligibility rule is, that kids must not work more than 20
    hour and are below 25.

    """
    return (alter < altersgrenze["ohne_bedingungen"]) or (
        (alter < altersgrenze["mit_bedingungen"])
        and in_ausbildung
        and (arbeitsstunden_w <= maximale_arbeitsstunden_des_kindes)
    )


@policy_function()
def kind_bis_10_mit_kindergeld(
    alter: int,
    grundsätzlich_anspruchsberechtigt: bool,
) -> bool:
    """Child under the age of 11 and eligible for Kindergeld.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    grundsätzlich_anspruchsberechtigt
        See :func:`grundsätzlich_anspruchsberechtigt_nach_stunden`.

    Returns
    -------

    """
    return grundsätzlich_anspruchsberechtigt and (alter <= 10)


@policy_function(vectorization_strategy="not_required")
def gleiche_fg_wie_empfänger(
    p_id: np.ndarray,  # int
    p_id_empfänger: np.ndarray,  # int
    fg_id: np.ndarray,  # int
) -> np.ndarray:  # bool
    """The child's Kindergeldempfänger is in the same Familiengemeinschaft.

    Parameters
    ----------
    p_id
        See basic input variable :ref:`p_id <p_id>`.
    p_id_empfänger
        See basic input variable :ref:`p_id_empfänger <p_id_empfänger>`.
    fg_id
        See basic input variable :ref:`fg_id <fg_id>`.

    Returns
    -------

    """
    fg_id_kindergeldempfänger = join(
        p_id_empfänger,
        p_id,
        fg_id,
        value_if_foreign_key_is_missing=-1,
    )

    return fg_id_kindergeldempfänger == fg_id
