from ttsim import FKType, policy_input


@policy_input()
def p_id() -> int:
    """Person ID, always required by TTSIM."""


@policy_input()
def hh_id() -> int:
    """Household id (will delete once fam_id is enough)."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_parent_1() -> int:
    """Identifier of the first parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_parent_2() -> int:
    """Identifier of the second parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_spouse() -> int:
    """Identifier of married partner."""


@policy_input()
def age() -> int:
    """Age of the person."""
