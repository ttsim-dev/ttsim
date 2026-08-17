"""Tests for unit declaration validation and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import numpy
import pytest

from tests.test_unit_fixtures import (
    _END,
    _START,
    CASTAR,
    CASTAR_PER_MONTH,
    CASTAR_PER_YEAR,
    GROUPING_LEVELS,
    REGISTRY,
    UNIT_SYSTEM,
    _scalar_unit,
    _unit_tree,
    amount_y,
    bonus_y,
    fam_id,
    is_exempt,
    make_flow_rate,
    p_id,
    p_id_recipient,
    statutory_age,
    unannotated_income_y,
    wealth,
)
from ttsim.exceptions import (
    AggregationDefinitionError,
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
)
from ttsim.tt import (
    UNSET_UNIT,
    AggType,
    RoundingSpec,
    TTSIMUnit,
    agg_by_group_function,
    cast_ttsim_unit,
    param_function,
    policy_function,
    policy_input,
)
from ttsim.tt.param_objects import (
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    ScalarParam,
)
from ttsim.tt.units import (
    divide_by_grouping_level,
    pint_unit_from_string,
    units_are_equivalent,
)
from ttsim.unit_resolution import (
    _structured_field_kinds,
    node_is_boolean,
    resolve_environment_units,
)
from ttsim.unit_validation import (
    create_unit_validation_report,
    fail_if_environment_units_are_inconsistent,
    fail_if_environment_units_are_missing,
)

# Mandatory units, no exemptions: identifiers and booleans declare
# DIMENSIONLESS; group-creation group ids are auto-assigned DIMENSIONLESS;
# framework date nodes get their unit from the framework.


def test_boolean_nodes_are_detected():
    assert node_is_boolean(qname="is_exempt", obj=is_exempt)
    assert not node_is_boolean(qname="wealth", obj=wealth)


def test_group_marker_on_a_float_share_is_rejected():
    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def housing_cost_share_fam() -> float:
        """The household's housing-cost share."""

    with pytest.raises(UnitConsistencyError, match=r"share|count or yes/no"):
        fail_if_environment_units_are_inconsistent(
            env={"housing_cost_share_fam": housing_cost_share_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_group_marker_on_an_integer_category_is_rejected():
    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def rent_class_fam() -> int:
        """The municipality's rent classification."""

    with pytest.raises(UnitConsistencyError, match="count or yes/no"):
        fail_if_environment_units_are_inconsistent(
            env={"rent_class_fam": rent_class_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_documented_integer_count_may_carry_a_group_marker():
    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def number_of_children_fam() -> int:
        """Number of children in the family."""

    fail_if_environment_units_are_inconsistent(
        env={"number_of_children_fam": number_of_children_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_boolean_may_carry_a_group_marker():
    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def eligible_fam() -> bool:
        """Whether the family is eligible."""

    fail_if_environment_units_are_inconsistent(
        env={"eligible_fam": eligible_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_validation_report_separates_evidence_and_exceptions():
    @policy_input(unit=TTSIMUnit.CURRENCY)
    def assets() -> float:
        """Financial assets."""

    @policy_function(unit=TTSIMUnit.CURRENCY)
    def checked_assets(assets: float) -> float:
        return assets

    @policy_function(unit=TTSIMUnit.CURRENCY)
    def locally_asserted_assets(assets: float) -> float:
        return cast_ttsim_unit(assets, unit=TTSIMUnit.CURRENCY)

    @policy_function(unit=TTSIMUnit.CURRENCY, verify_units=False)
    def opted_out_assets(assets: float) -> float:
        return assets

    @agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def number_of_people_fam(fam_id: int) -> int: ...

    env = {
        "assets": assets,
        "checked_assets": checked_assets,
        "locally_asserted_assets": locally_asserted_assets,
        "opted_out_assets": opted_out_assets,
        "number_of_people_fam": number_of_people_fam,
    }
    report = create_unit_validation_report(
        env=env,
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
        policy_dates=(_START,),
    )

    assert report.resolved_declarations == tuple(sorted(env))
    assert report.checked_function_bodies == (
        "checked_assets",
        "locally_asserted_assets",
    )
    assert report.generated_rules == ("number_of_people_fam",)
    assert report.local_casts == ("locally_asserted_assets",)
    assert report.body_opt_outs == ("opted_out_assets",)
    assert report.unsupported_bodies == ()
    assert report.other_unchecked_bodies == ()
    assert report.policy_date_regimes == (_START,)


def test_validation_report_names_a_body_rejected_by_inference():
    @policy_input(unit=TTSIMUnit.CURRENCY)
    def assets() -> float:
        """Financial assets."""

    @policy_function(unit=TTSIMUnit.YEARS)
    def wrong_unit(assets: float) -> float:
        return assets

    report = create_unit_validation_report(
        env={"assets": assets, "wrong_unit": wrong_unit},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )

    assert report.checked_function_bodies == ()
    assert len(report.unsupported_bodies) == 1
    assert report.unsupported_bodies[0].qname == "wrong_unit"
    assert "declares" in report.unsupported_bodies[0].reason


def test_missing_check_passes_for_declared_and_group_creation_nodes():
    fail_if_environment_units_are_missing(
        {
            # Identifiers and the boolean declare DIMENSIONLESS (GEP 10);
            # the group-creation `fam_id` is auto-assigned DIMENSIONLESS.
            "p_id": p_id,
            "p_id_recipient": p_id_recipient,
            "fam_id": fam_id,
            "is_exempt": is_exempt,
            "wealth": wealth,
            "tax_rate_y": make_flow_rate(),
            "amount_y": amount_y,
        }
    )


def test_missing_check_reports_unannotated_node():
    with pytest.raises(UnitDefinitionError, match="unannotated_income_y"):
        fail_if_environment_units_are_missing(
            {"unannotated_income_y": unannotated_income_y}
        )


def test_missing_check_reports_unannotated_identifier_and_boolean():
    # Identifiers and booleans declare `DIMENSIONLESS` like every other node
    # (GEP 10): an undeclared one is reported, whatever its data type.
    @policy_input(unit=UNSET_UNIT)
    def some_id() -> int:
        """An identifier carrying the UNSET sentinel."""

    @policy_input(unit=UNSET_UNIT)
    def some_flag() -> bool:
        """A boolean carrying the UNSET sentinel."""

    with pytest.raises(UnitDefinitionError, match=r"(?s)some_flag.*some_id"):
        fail_if_environment_units_are_missing(
            {"some_id": some_id, "some_flag": some_flag}
        )


# Currency-denominated rounding specs: mandatory on a currency-valued function,
# forbidden elsewhere, composite must equal the function's declared unit with
# the agnostic base swapped for the concrete currency (GEP 10)


def make_rounded_amount_y(rounding_spec: RoundingSpec):
    @policy_function(rounding_spec=rounding_spec, unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def rounded_amount_y(bonus_y: float) -> float:
        return bonus_y

    return rounded_amount_y


def test_missing_check_reports_currency_rounding_spec_without_unit():
    with pytest.raises(
        UnitDefinitionError, match=r"rounded_amount_y \(rounding_spec\)"
    ):
        fail_if_environment_units_are_missing(
            {
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down")
                ),
                "bonus_y": bonus_y,
            }
        )


def test_missing_check_passes_for_currency_rounding_spec_with_unit():
    fail_if_environment_units_are_missing(
        {
            "rounded_amount_y": make_rounded_amount_y(
                RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR)
            ),
            "bonus_y": bonus_y,
        }
    )


def test_missing_check_passes_for_non_currency_rounding_spec_without_unit():
    @policy_function(
        rounding_spec=RoundingSpec(base=1, direction="down"), unit=TTSIMUnit.YEARS
    )
    def rounded_age(statutory_age: int) -> int:
        return statutory_age

    fail_if_environment_units_are_missing(
        {"rounded_age": rounded_age, "statutory_age": statutory_age}
    )


def test_inconsistency_check_passes_for_matching_rounding_spec_unit():
    fail_if_environment_units_are_inconsistent(
        env={
            "rounded_amount_y": make_rounded_amount_y(
                RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR)
            ),
            "bonus_y": bonus_y,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_inconsistency_check_reports_rounding_spec_composite_mismatch():
    with pytest.raises(UnitConsistencyError, match="same flow period"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down", unit=CASTAR_PER_MONTH)
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_inconsistency_check_reports_agnostic_rounding_spec_unit():
    with pytest.raises(UnitConsistencyError, match="concrete"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(
                        base=1, direction="down", unit=TTSIMUnit.CURRENCY.PER_YEAR
                    )
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_inconsistency_check_reports_non_currency_rounding_spec_unit():
    with pytest.raises(UnitConsistencyError, match="registered currency"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down", unit=TTSIMUnit.YEARS)
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_inconsistency_check_reports_rounding_spec_unit_on_non_currency_function():
    @policy_function(
        rounding_spec=RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR),
        unit=TTSIMUnit.YEARS,
    )
    def rounded_age(statutory_age: int) -> int:
        return statutory_age

    with pytest.raises(UnitConsistencyError, match="nothing to convert"):
        fail_if_environment_units_are_inconsistent(
            env={"rounded_age": rounded_age, "statutory_age": statutory_age},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Unit resolution


def test_resolution_combines_token_and_name_suffix():
    resolved = resolve_environment_units(
        env={
            "wealth": wealth,
            "tax_rate_y": make_flow_rate(),
            "amount_y": amount_y,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    # `wealth` is declared with a bare currency unit at an unsuffixed name, so it
    # is level-neutral (GEP 10) — it carries no grouping level; `tax_rate_y` is a
    # bare rate and likewise level-neutral.
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="wealth"),
        right=pint_unit_from_string(unit_str="CURRENCY", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="tax_rate_y"),
        right=pint_unit_from_string(unit_str="1 / year", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="amount_y"),
        right=pint_unit_from_string(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_resolution_includes_framework_date_nodes():
    # `policy_year` is a calendar *point*, not a duration (GEP 10): it is not
    # equivalent to a `year` duration.
    env = {
        "policy_year": ScalarParam(value=2020, start_date=_START, end_date=_END),
    }
    resolved = resolve_environment_units(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="policy_year"),
        right=pint_unit_from_string(unit_str="calendar_year", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert not units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="policy_year"),
        right=pint_unit_from_string(unit_str="year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_dict_param_with_per_leaf_units_resolves_to_unit_tree():
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    unit_tree = _unit_tree(resolved=resolved, qname="schedule")
    assert units_are_equivalent(
        left=unit_tree["child_amount_y"],
        right=pint_unit_from_string(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=unit_tree["max_age"],
        right=pint_unit_from_string(unit_str="year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_dict_param_leaf_key_suffix_must_agree_with_spelled_period():
    # A leaf key's time suffix must agree with the period spelled in its unit
    # (GEP 10): a `_y` key declaring a per-month unit is a contradiction.
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CASTAR_PER_MONTH"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"schedule": schedule},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dict_param_integer_keyed_flow_leaf_spells_its_period():
    # Integer keys cannot carry a suffix; the leaf's unit spells the period
    # directly (GEP 10).
    amount_by_rank = DictParam(
        value={1: 250.0, 2: 250.0},
        unit={1: "CASTAR_PER_MONTH", 2: "CASTAR_PER_MONTH"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"amount_by_rank": amount_by_rank},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    assert units_are_equivalent(
        left=_unit_tree(resolved=resolved, qname="amount_by_rank")[1],
        right=pint_unit_from_string(unit_str="CURRENCY / month", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_dict_param_stock_token_on_suffixed_leaf_key_fails():
    schedule = DictParam(
        value={"amount_y": 100.0},
        unit={"amount_y": "CASTAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"schedule": schedule},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dict_param_missing_leaf_unit_is_reported():
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match=r"schedule\[max_age\]"):
        fail_if_environment_units_are_missing({"schedule": schedule})


def test_scalar_flow_param_resolves_via_name_suffix():
    lump_sum = ScalarParam(
        value=100.0,
        unit=CASTAR_PER_YEAR,
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"lump_sum_deduction_y": lump_sum},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="lump_sum_deduction_y"),
        right=pint_unit_from_string(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_scalar_param_spelled_period_must_agree_with_name_suffix():
    # The spelled period must agree with the name's time suffix (GEP 10): a
    # stock CASTAR on a `_y` name is a contradiction.
    threshold = ScalarParam(
        value=100.0,
        unit=CASTAR,
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"some_amount_y": threshold},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_group_sum_of_a_head_count_source_derives_head_count_per_group():
    """Summing a bare per-person head count to a group derives
    ``DIMENSIONLESS_PER_<group>`` (``1/[group]``), so the derivation matches the
    minted token and a valid declaration passes (GEP 10)."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def n_children() -> int:
        """A per-person head count."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def n_children_fam(n_children: int, fam_id: int) -> int:
        """The family's total head count."""

    fail_if_environment_units_are_inconsistent(
        env={"n_children": n_children, "n_children_fam": n_children_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_aggregation_must_spell_the_derived_grouping_level():
    """An aggregation's declared unit must be precise and complete: a ``_fam`` sum
    declaring a bare ``CURRENCY`` (omitting the derived ``[fam]`` level) is rejected
    — there is no implicit matching of group levels, the author spells it (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.CURRENCY)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """A family sum that fails to spell its [fam] level."""

    with pytest.raises(UnitConsistencyError, match="wealth_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "wealth_fam": wealth_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_aggregation_with_the_precise_derived_unit_passes():
    """A precise declaration — kind, period, level, agnostic currency — passes.

    An aggregation is a column, so it declares the agnostic ``CURRENCY`` and its
    declared unit is checked against the derivation.
    """

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.CURRENCY.PER_FAM)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """Family wealth, level spelled."""

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "wealth_fam": wealth_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_aggregation_with_spelled_wrong_grouping_level_is_caught():
    """A spelled grouping level that contradicts the derivation is rejected: a
    ``_fam`` sum declaring ``CURRENCY_PER_KIN`` derives ``[fam]`` (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.CURRENCY.PER_KIN)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """A family sum mis-declared at the kin level."""

    with pytest.raises(UnitConsistencyError, match="wealth_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "wealth_fam": wealth_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_aggregation_declaring_a_concrete_currency_is_rejected():
    """An aggregation runs in the statutory currency of the policy date, so pinning
    a concrete currency down is a definition error."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.CASTAR.PER_FAM)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """A family sum wrongly pinning CASTAR."""

    with pytest.raises(UnitDefinitionError, match="agnostic CURRENCY"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "wealth_fam": wealth_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_aggregation_decorator_rejects_invalid_unit():
    # Strings are not tokens: the decorator's type contract only admits
    # `TTSIMUnit` members (or None), enforced by the beartype claw.
    with pytest.raises(AggregationDefinitionError, match="unit"):

        @agg_by_group_function(agg_type=AggType.SUM, unit="kelvin")  # ty: ignore[invalid-argument-type]
        def bad_fam(wealth: float, fam_id: int) -> float:
            """Invalid unit."""


# Param functions


def test_param_function_unit_resolves_via_leaf_name_suffix():
    @param_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def max_amount_m_fam(policy_year: int) -> float:
        return float(policy_year)

    resolved = resolve_environment_units(
        env={"max_amount_m_fam": max_amount_m_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    # The `_fam` suffix puts this flow at the family level (GEP 10).
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="max_amount_m_fam"),
        right=pint_unit_from_string(
            unit_str="CURRENCY / month / grouping_level_fam", registry=REGISTRY
        ),
        registry=REGISTRY,
    )


# Parameters must pin down their concrete currency (GEP 10)


def test_scalar_param_with_agnostic_currency_token_fails():
    threshold = ScalarParam(
        value=100.0,
        unit=TTSIMUnit.CURRENCY,
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={"threshold": threshold},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dict_param_leaf_with_agnostic_currency_token_fails():
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CURRENCY_PER_YEAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={"schedule": schedule},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_concrete_currency_token_resolves_like_agnostic_counterpart():
    # Union semantics: the concrete currency drives only the build-time
    # conversion, never the dimensionality.
    threshold = ScalarParam(
        value=100.0,
        unit=CASTAR,
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"threshold": threshold},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="threshold"),
        right=pint_unit_from_string(unit_str="CURRENCY", registry=REGISTRY),
        registry=REGISTRY,
    )


# Mapping parameters declare per-axis units (GEP 10)


def _make_schedule_param(**kwargs: Any) -> PiecewisePolynomialParam:
    return PiecewisePolynomialParam(
        value=PiecewisePolynomialParamValue(
            thresholds=numpy.array([0.0, 1.0]),
            intercepts=numpy.array([0.0]),
            coefficients=numpy.array([[0.1]]),
        ),
        start_date=_START,
        end_date=_END,
        **kwargs,
    )


def test_param_mapping_object_rejects_unit_declaration():
    with pytest.raises(UnitDefinitionError, match=r"input_unit.*instead of"):
        _make_schedule_param(unit=CASTAR_PER_YEAR)


def test_param_mapping_object_resolves_output_axis():
    # An income schedule: both axes are currency flows, each spelling its period;
    # the output is a per-person amount (GEP 10).
    schedule = _make_schedule_param(
        input_unit=CASTAR_PER_YEAR,
        output_unit=CASTAR_PER_YEAR,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=pint_unit_from_string(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_param_mapping_object_complete_input_axis_with_flow_output():
    # The property-tax shape: hectares in, a per-person yearly currency flow out.
    schedule = _make_schedule_param(
        input_unit=TTSIMUnit.HECTARE,
        output_unit=CASTAR_PER_YEAR,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=pint_unit_from_string(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_param_mapping_object_rejects_agnostic_currency_axis():
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={
                "schedule": _make_schedule_param(
                    input_unit=TTSIMUnit.CURRENCY.PER_YEAR,
                    output_unit=CASTAR_PER_YEAR,
                )
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_param_mapping_object_missing_axis_units_are_reported():
    with pytest.raises(UnitDefinitionError) as excinfo:
        fail_if_environment_units_are_missing({"schedule": _make_schedule_param()})
    assert "schedule (input_unit)" in str(excinfo.value)
    assert "schedule (output_unit)" in str(excinfo.value)


def test_auto_generated_boolean_group_aggregate_passes_the_build():
    """Requesting the group aggregate of a boolean auto-generates a SUM node, whose
    framework-minted token must match what the resolver derives (a head count,
    1/[fam]). The build accepts its own auto-assignment (GEP 10)."""

    @policy_function(leaf_name="is_adult", unit=TTSIMUnit.DIMENSIONLESS)
    def is_adult() -> bool:
        return True

    aggs = create_agg_by_group_functions(
        column_functions={"is_adult": is_adult},
        qname_policy_environment={},
        time_converted_input_stubs={},
        data_qnames=set(),
        tt_targets={"is_adult_fam"},
        grouping_levels=("fam",),
    )
    # No UnitConsistencyError: the minted token and the derived unit agree.
    fail_if_environment_units_are_inconsistent(
        env={"is_adult": is_adult, "is_adult_fam": aggs.functions["is_adult_fam"]},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_group_mean_derives_the_target_level_without_an_opt_out():
    """A group mean is a statistic of the target group (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> float: ...

    @agg_by_group_function(agg_type=AggType.MEAN, unit=TTSIMUnit.CURRENCY.PER_KIN)
    def average_wealth_kin(kin_id: int, wealth: float) -> float: ...

    env = {"wealth": wealth, "average_wealth_kin": average_wealth_kin}
    resolved = resolve_environment_units(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="average_wealth_kin"),
        right=pint_unit_from_string(
            unit_str="CURRENCY / grouping_level_kin", registry=REGISTRY
        ),
        registry=REGISTRY,
    )
    fail_if_environment_units_are_inconsistent(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )


def test_dimensionless_group_mean_is_validated_as_an_aggregation():
    """A derived group statistic is not a direct group-quantity declaration."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def share() -> float: ...

    @agg_by_group_function(
        agg_type=AggType.MEAN,
        unit=TTSIMUnit.DIMENSIONLESS.PER_KIN,
    )
    def average_share_kin(kin_id: int, share: float) -> float: ...

    fail_if_environment_units_are_inconsistent(
        env={"share": share, "average_share_kin": average_share_kin},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_group_mean_declared_bare_is_rejected():
    """A group mean cannot be declared as a person-level result (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY)
    def wealth() -> float: ...

    @agg_by_group_function(agg_type=AggType.MEAN, unit=TTSIMUnit.CURRENCY)
    def average_wealth_kin(kin_id: int, wealth: float) -> float: ...

    with pytest.raises(UnitConsistencyError, match="average_wealth_kin"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "average_wealth_kin": average_wealth_kin},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_count_and_sum_of_boolean_both_mint_head_counts():
    """A COUNT and a SUM over a boolean are both head counts (GEP 10): each
    resolves to a dimensionless 1/[target], not a bare DIMENSIONLESS."""

    @agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def number_of_individuals_fam(fam_id: int) -> int: ...

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def is_adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS)
    def number_of_adults_fam(fam_id: int, is_adult: bool) -> int: ...

    resolved = resolve_environment_units(
        env={
            "number_of_individuals_fam": number_of_individuals_fam,
            "is_adult": is_adult,
            "number_of_adults_fam": number_of_adults_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )
    head_count = divide_by_grouping_level(
        unit=REGISTRY.dimensionless, level="fam", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="number_of_individuals_fam"),
        right=head_count,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="number_of_adults_fam"),
        right=head_count,
        registry=REGISTRY,
    )


def test_per_capita_division_bridges_via_head_count():
    """A group total divided by a head count type-checks to a bare per-person
    amount: (CURRENCY/[fam]) / (1/[fam]) = CURRENCY (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def number_of_individuals_fam(fam_id: int) -> int: ...

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def rent_m_fam() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def rent_per_head_m(rent_m_fam: float, number_of_individuals_fam: int) -> float:
        return rent_m_fam / number_of_individuals_fam

    env = {
        "number_of_individuals_fam": number_of_individuals_fam,
        "rent_m_fam": rent_m_fam,
        "rent_per_head_m": rent_per_head_m,
    }
    resolved = resolve_environment_units(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="rent_per_head_m"),
        right=pint_unit_from_string(unit_str="CURRENCY / month", registry=REGISTRY),
        registry=REGISTRY,
    )
    # The [fam] cancels against the count's 1/[fam] — no level mismatch.
    fail_if_environment_units_are_inconsistent(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )


# Aggregations: declared unit must match the derived unit (GEP 10)


def test_sum_over_boolean_declared_bare_dimensionless_is_caught():
    """A SUM over a boolean is a head count, so it belongs to the group it counts
    within: it derives `1/[fam]` (the persons the indicator is true for). The
    declaration must spell that level — `DIMENSIONLESS_PER_FAM`, not the bare
    `DIMENSIONLESS` (GEP 10).
    """

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS)
    def number_of_adults_fam(adult: bool, fam_id: int) -> int: ...

    with pytest.raises(UnitConsistencyError, match="number_of_adults_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"adult": adult, "number_of_adults_fam": number_of_adults_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_sum_over_boolean_declared_head_count_per_group_passes():
    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def number_of_adults_fam(adult: bool, fam_id: int) -> int: ...

    fail_if_environment_units_are_inconsistent(
        env={"adult": adult, "number_of_adults_fam": number_of_adults_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_sum_of_currency_declared_with_wrong_kind_is_caught():
    """A SUM of a currency flow derives currency; declaring YEARS is rejected."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.YEARS)
    def income_m_fam(income_m: float, fam_id: int) -> float: ...

    with pytest.raises(UnitConsistencyError, match="income_m_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "income_m_fam": income_m_fam},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_max_over_bare_source_carries_the_target_group_level():
    """Aggregations follow the *base*, not the agg type: a MAX of a bare person
    income acquires the target group level like a SUM. A `_fam` MAX is
    CURRENCY/month/[fam], not the source's bare level, and is declared `..._PER_FAM`.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @agg_by_group_function(
        agg_type=AggType.MAX, unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM
    )
    def income_max_m_fam(income_m: float, fam_id: int) -> float: ...

    env = {"income_m": income_m, "income_max_m_fam": income_max_m_fam}
    resolved = resolve_environment_units(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )
    max_unit = _scalar_unit(resolved=resolved, qname="income_max_m_fam")
    # The MAX carries the target [fam] level, not the source's bare level.
    assert units_are_equivalent(
        left=max_unit,
        right=divide_by_grouping_level(
            unit=pint_unit_from_string(unit_str="CURRENCY / month", registry=REGISTRY),
            level="fam",
            registry=REGISTRY,
        ),
        registry=REGISTRY,
    )
    assert not units_are_equivalent(
        left=max_unit,
        right=pint_unit_from_string(unit_str="CURRENCY / month", registry=REGISTRY),
        registry=REGISTRY,
    )
    # The `_PER_FAM` declaration is consistent with what it derives.
    fail_if_environment_units_are_inconsistent(
        env=env, grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )


def test_structured_field_kinds_skips_only_the_unresolvable_field():
    """A field whose annotation names something visible only to type checkers stays
    opaque, while a sibling whose annotation nests a forward reference inside
    ``Annotated`` still resolves to its unit."""

    @dataclass
    class SpecWithOneUnresolvableField:
        rate: Annotated[float, TTSIMUnit.DIMENSIONLESS]
        nested_forward_ref: Annotated["float", TTSIMUnit.DIMENSIONLESS]  # noqa: UP037
        opaque: OnlyVisibleToTypeCheckers  # noqa: F821  # ty: ignore[unresolved-reference]

    kinds = _structured_field_kinds(
        cls=SpecWithOneUnresolvableField, unit_system=UNIT_SYSTEM
    )
    assert kinds == {
        "rate": REGISTRY.dimensionless,
        "nested_forward_ref": REGISTRY.dimensionless,
    }


def test_structured_field_kinds_keeps_inherited_annotations():
    """A subclass with one unresolvable annotation keeps the resolved units of the
    fields it inherits."""

    @dataclass
    class BaseSpec:
        rate: Annotated[float, TTSIMUnit.DIMENSIONLESS]

    @dataclass
    class DerivedSpec(BaseSpec):
        opaque: AlsoOnlyVisibleToTypeCheckers  # noqa: F821  # ty: ignore[unresolved-reference]

    kinds = _structured_field_kinds(cls=DerivedSpec, unit_system=UNIT_SYSTEM)
    assert kinds == {"rate": REGISTRY.dimensionless}
