import optree
import pandas as pd
import pytest

from _gettsim.interface import oss


@pytest.fixture
def example_inputs_df():
    return pd.DataFrame(
        {
            "id": [0, 1, 2],
            "gross_wage": [2000, 0, 0],
            "age": [20, 2, 2],
            "birth_year": [1990, 2023, 2023],
            "pointer_parent_1": [-1, 0, 0],
            "recipient_child_benefits_id": [-1, 0, 0],
            "is_single_parent": [True, False, False],
            "has_children": [True, False, False],
        }
    )


@pytest.fixture
def example_inputs_tree_to_inputs_df_columns():
    return {
        "arbeitsstunden_w": 0,
        "einkommensteuer": {
            "abzüge": {
                "kinderbetreuungskosten_m": 0.0,
                "p_id_kinderbetreuungskostenträger": -1,
                "beitrag_private_rentenversicherung_m": 0.0,
            },
            "einkünfte": {
                "ist_selbstständig": False,
                "aus_nichtselbstständiger_arbeit": {
                    "bruttolohn_m": "gross_wage",
                },
                "aus_selbstständiger_arbeit": {
                    "betrag_m": 0.0,
                },
                "aus_gewerbebetrieb": {
                    "betrag_m": 0.0,
                },
                "aus_forst_und_landwirtschaft": {
                    "betrag_m": 0.0,
                },
                "aus_kapitalvermögen": {
                    "kapitalerträge_m": 0.0,
                },
                "aus_vermietung_und_verpachtung": {
                    "betrag_m": 0.0,
                },
                "sonstige": {
                    "ohne_renten_m": 0.0,
                },
            },
            "gemeinsam_veranlagt": False,
        },
        "alter": "age",
        "behinderungsgrad": 0,
        "p_id": "id",
        "geburtsjahr": "birth_year",
        "familie": {
            "p_id_ehepartner": -1,
            "p_id_elternteil_1": "pointer_parent_1",
            "p_id_elternteil_2": -1,
            "alleinerziehend": "is_single_parent",
        },
        "kindergeld": {
            "p_id_empfänger": "recipient_child_benefits_id",
            "in_ausbildung": False,
        },
        "sozialversicherung": {
            "rente": {
                "altersrente": {
                    "betrag_m": 0.0,
                },
                "jahr_renteneintritt": 2060,
                "private_rente_betrag_m": 0.0,
            },
            "pflege": {
                "beitrag": {
                    "hat_kinder": "has_children",
                },
            },
            "kranken": {
                "beitrag": {
                    "privat_versichert": False,
                }
            },
        },
        "wohnort_ost": False,
    }


_EXAMPLE_TARGETS_TREE_TO_DF_COLUMNS = {
    "einkommensteuer": {
        "betrag_y_sn": "income_tax",  # policy target
        "kinderfreibetrag_pro_kind_y": "child_tax_credit_per_child",  # param target
    },
}


@pytest.mark.parametrize(
    "targets_tree_to_outputs_df_columns",
    [
        # Param target and policy target
        {
            "einkommensteuer": {
                "betrag_y_sn": "income_tax",
                "kinderfreibetrag_pro_kind_y": "child_tax_credit_per_child",
            },
        },
        # Policy target only
        {
            "einkommensteuer": {
                "betrag_y_sn": "income_tax",
            },
        },
        # Param target only
        {
            "einkommensteuer": {
                "kinderfreibetrag_pro_kind_y": "child_tax_credit_per_child",
            },
        },
    ],
)
def test_oss_with_gettsim_policy_env(
    targets_tree_to_outputs_df_columns,
    example_inputs_df,
    example_inputs_tree_to_inputs_df_columns,
):
    results = oss(
        date="2024-01-01",
        inputs_df=example_inputs_df,
        inputs_tree_to_inputs_df_columns=example_inputs_tree_to_inputs_df_columns,
        targets_tree_to_outputs_df_columns=targets_tree_to_outputs_df_columns,
    )
    expected_columns: list[tuple[str]] = optree.tree_flatten(
        targets_tree_to_outputs_df_columns
    )[0]
    assert results.shape == (
        example_inputs_df.shape[0],
        len(expected_columns),
    )
    assert all(col in results.columns for col in expected_columns)
