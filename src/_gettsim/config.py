from __future__ import annotations

import importlib
from pathlib import Path

import numpy

# Defaults
USE_JAX = False
numpy_or_jax = numpy


def set_array_backend(backend: str):
    """Set array library backend.

    backend (str): Must be in {'jax', 'numpy'}.

    """
    if backend not in {"jax", "numpy"}:
        raise ValueError(f"Backend must be in {'jax', 'numpy'} but is {backend}.")

    if backend == "jax":
        assert importlib.util.find_spec("jax") is not None, "JAX is not installed."
        global USE_JAX  # noqa: PLW0603
        global numpy_or_jax  # noqa: PLW0603
        import jax

        USE_JAX = True
        numpy_or_jax = jax.numpy
        jax.config.update("jax_platform_name", "cpu")


# Obtain the root directory of the package.
RESOURCE_DIR = Path(__file__).parent.resolve()

GEP_01_CHARACTER_LIMIT_USER_FACING_COLUMNS = 20
GEP_01_CHARACTER_LIMIT_OTHER_COLUMNS = 32


# List of paths to internal functions.
# If a path is a directory, all Python files are recursively collected from that folder.
PATHS_TO_INTERNAL_FUNCTIONS = [
    RESOURCE_DIR / "transfers",
    RESOURCE_DIR / "taxes",
]

INTERNAL_PARAMS_GROUPS = [
    "eink_st",
    "eink_st_abzuege",
    "soli_st",
    "arbeitsl_geld",
    "sozialv_beitr",
    "unterhalt",
    "unterhaltsvors",
    "abgelt_st",
    "wohngeld",
    "kinderzuschl",
    "kinderzuschl_eink",
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


SUPPORTED_TIME_UNITS = {
    "y": {
        "name": "year",
    },
    "m": {
        "name": "month",
    },
    "w": {
        "name": "week",
    },
    "d": {
        "name": "day",
    },
}

DEFAULT_TARGETS = {
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


TYPES_INPUT_VARIABLES = {
    "arbeitslosengeld_2": {
        "arbeitslosengeld_2_bezug_im_vorjahr": bool,
        # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt once
        # Bedarfsgemeinschaften are fully endogenous
        # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763
        "eigenbedarf_gedeckt": bool,
        "p_id_einstandspartner": int,
    },
    "familie": {
        "alleinerziehend": bool,
        "kind": bool,
        "p_id_ehepartner": int,
        "p_id_elternteil_1": int,
        "p_id_elternteil_2": int,
    },
    "alter": int,
    "arbeitsstunden_w": float,
    "behinderungsgrad": int,
    "geburtsjahr": int,
    "geburtsmonat": int,
    "geburtstag": int,
    "schwerbehindert_grad_g": bool,
    "vermögen": float,
    "weiblich": bool,
    "wohnort_ost": bool,
    "einkommensteuer": {
        "abzüge": {
            "beitrag_private_rentenversicherung_m": float,
            "betreuungskosten_m": float,
            "p_id_betreuungskosten_träger": int,
        },
        "einkünfte": {
            "aus_kapitalvermögen": {
                "kapitalerträge_m": float,
            },
            "aus_nichtselbstständiger_arbeit": {
                "bruttolohn_m": float,
                "bruttolohn_vorjahr_m": float,
            },
            "aus_selbstständiger_arbeit": {
                "betrag_m": float,
            },
            "aus_vermietung_und_verpachtung": {
                "betrag_m": float,
            },
            "ist_selbstständig": bool,
            "sonstige": {
                "betrag_m": float,
            },
        },
        "gemeinsam_veranlagt": bool,
    },
    "elterngeld": {
        "bisherige_bezugsmonate": int,
        "claimed": bool,
        "nettoeinkommen_vorjahr_m": float,
        "zu_versteuerndes_einkommen_vorjahr_y_sn": float,
    },
    "erziehungsgeld": {
        "budgetsatz": bool,
        "p_id_empfänger": int,
    },
    "hh_id": int,
    "kindergeld": {
        "in_ausbildung": bool,
        "p_id_empfänger": int,
    },
    "lohnsteuer": {
        "steuerklasse": int,
    },
    "p_id": int,
    "sozialversicherung": {
        "arbeitslosen": {
            "anwartschaftszeit": bool,
            "arbeitssuchend": bool,
            "monate_durchgängigen_bezugs_von_arbeitslosengeld": float,
            "monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren": float,
        },
        "kranken": {
            "beitrag": {
                "privat_versichert": bool,
            }
        },
        "pflege": {
            "beitrag": {
                "hat_kinder": bool,
            }
        },
        "rente": {
            "altersrente": {
                "für_frauen": {
                    "pflichtsbeitragsjahre_ab_alter_40": float,
                },
                "höchster_bruttolohn_letzte_15_jahre_vor_rente_y": float,
                "wegen_arbeitslosigkeit": {
                    "arbeitslos_für_1_jahr_nach_alter_58_ein_halb": bool,
                    "pflichtbeitragsjahre_8_von_10": bool,
                    "vertrauensschutz_1997": bool,
                    "vertrauensschutz_2004": bool,
                },
            },
            "bezieht_rente": bool,
            "entgeltpunkte_ost": float,
            "entgeltpunkte_west": float,
            "erwerbsminderung": {
                "teilweise_erwerbsgemindert": bool,
                "voll_erwerbsgemindert": bool,
            },
            "ersatzzeiten_monate": float,
            "freiwillige_beitragsmonate": float,
            "grundrente": {
                "bewertungszeiten_monate": int,
                "grundrentenzeiten_monate": int,
                "mean_entgeltpunkte": float,
            },
            "jahr_renteneintritt": int,
            "kinderberücksichtigungszeiten_monate": float,
            "krankheitszeiten_ab_16_bis_24_monate": float,
            "monat_renteneintritt": int,
            "monate_geringfügiger_beschäftigung": float,
            "monate_in_arbeitslosigkeit": float,
            "monate_in_arbeitsunfähigkeit": float,
            "monate_in_ausbildungssuche": float,
            "monate_in_mutterschutz": float,
            "monate_in_schulausbildung": float,
            "monate_mit_bezug_entgeltersatzleistungen_wegen_arbeitslosigkeit": float,
            "pflichtbeitragsmonate": float,
            "private_rente_betrag_m": float,
            "pflegeberücksichtigungszeiten_monate": float,
        },
    },
    "unterhalt": {
        "anspruch_m": float,
        "tatsächlich_erhaltener_betrag_m": float,
    },
    "wohngeld": {
        "mietstufe": int,
    },
    "wohnen": {
        "baujahr_immobilie_hh": int,
        "bewohnt_eigentum_hh": bool,
        "bruttokaltmiete_m_hh": float,
        "heizkosten_m_hh": float,
        "wohnfläche_hh": float,
    },
}

FOREIGN_KEYS = [
    ("arbeitslosengeld_2", "p_id_einstandspartner"),
    ("familie", "p_id_ehepartner"),
    ("familie", "p_id_elternteil_1"),
    ("familie", "p_id_elternteil_2"),
]
