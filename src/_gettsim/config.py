from pathlib import Path

# Obtain the root directory of the package.
GETTSIM_ROOT = Path(__file__).parent.resolve()

INTERNAL_PARAMS_GROUPS = [
    "geringfügige_einkommen",
    "unterhalt",
    "unterhaltsvors",
    "wohngeld",
    "kinderzuschl",
    "kindergeld",  # Leave because of _parse_kinderzuschlag_max
    "elterngeld",
    "erwerbsm_rente",
    "grunds_im_alter",
    "lohnst",
    "erziehungsgeld",
]

_TO_DELETE_DEFAULT_TARGETS = {
    "einkommensteuer": {
        "betrag_y_sn": None,
        "abgeltungssteuer": {"betrag_y_sn": None},
    },
    "solidaritätszuschlag": {"betrag_y_sn": None},
    "sozialversicherung": {
        "arbeitslosen": {
            "beitrag": {"betrag_versicherter_m": None},
            "betrag_m": None,
        },
        "kranken": {"beitrag": {"betrag_versicherter_m": None}},
        "pflege": {"beitrag": {"betrag_versicherter_m": None}},
        "rente": {
            "beitrag": {"betrag_versicherter_m": None},
            "altersrente": {"betrag_m": None},
            "erwerbsminderung": {"betrag_m": None},
        },
        "beiträge_versicherter_m": None,
    },
    "elterngeld": {"betrag_m": None},
    "kindergeld": {"betrag_m": None},
    "arbeitslosengeld_2": {"betrag_m_bg": None},
    "kinderzuschlag": {"betrag_m_bg": None},
    "wohngeld": {"betrag_m_wthh": None},
    "unterhaltsvorschuss": {"betrag_m": None},
    "grundsicherung": {"im_alter": {"betrag_m_eg": None}},
}
