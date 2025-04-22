from pathlib import Path

# Obtain the root directory of the package.
RESOURCE_DIR = Path(__file__).parent.resolve()

INTERNAL_PARAMS_GROUPS = [
    "eink_st",
    "eink_st_abzuege",
    "soli_st",
    "arbeitsl_geld",
    "sozialv_beitr",
    "arbeitslosenversicherung",
    "geringfügige_einkommen",
    "ges_krankenv",
    "ges_pflegev",
    "ges_rentenv",
    "unterhalt",
    "unterhaltsvors",
    "abgelt_st",
    "wohngeld",
    "kinderzuschl",
    "kindergeld",
    "elterngeld",
    "ges_rente",
    "erwerbsm_rente",
    "arbeitsl_geld_2",
    "grunds_im_alter",
    "lohnst",
    "erziehungsgeld",
]

SUPPORTED_GROUPINGS = {
    "hh": {
        "name": "Haushalt",
        "namespace": "top-level",
        "description": "Individuals living together in a household in the Wohngeld"
        " sense (§5 WoGG).",
        "potentially_endogenous": False,
    },
    "wthh": {
        "name": "wohngeldrechtlicher Teilhaushalt",
        "namespace": "wohngeld",
        "description": "The relevant unit for Wohngeld. Members of a household for whom"
        " the Wohngeld priority check compared to Bürgergeld yields the same result"
        " ∈ {True, False}.",
        "potentially_endogenous": True,
    },
    "fg": {
        "name": "Familiengemeinschaft",
        "namespace": "arbeitslosengeld_2",
        "description": "Maximum of two generations, the relevant base unit for"
        " Bürgergeld / Arbeitslosengeld 2, before excluding children who have enough"
        " income fend for themselves.",
        "potentially_endogenous": True,
    },
    "bg": {
        "name": "Bedarfsgemeinschaft",
        "namespace": "arbeitslosengeld_2",
        "description": "Familiengemeinschaft except for children who have enough income"
        " to fend for themselves. Relevant unit for Bürgergeld / Arbeitslosengeld 2",
        "potentially_endogenous": True,
    },
    "eg": {
        "name": "Einstandsgemeinschaft / Einstandspartner",
        "namespace": "arbeitslosengeld_2",
        "description": "A couple whose members are deemed to be responsible for each"
        " other.",
        "potentially_endogenous": True,
    },
    "ehe": {
        "name": "Ehepartner",
        "namespace": "familie",
        "description": "Couples that are either married or in a civil union.",
        "potentially_endogenous": True,
    },
    "sn": {
        "name": "Steuernummer",
        "namespace": "einkommensteuer",
        "description": "Spouses filing taxes jointly or individuals.",
        "potentially_endogenous": True,
    },
}


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
