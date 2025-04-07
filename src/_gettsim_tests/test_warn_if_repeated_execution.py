import pytest


def test_warn_when_internal_tests_are_executed_repeatedly():
    from gettsim import test

    test("--collect-only")

    with pytest.warns(UserWarning, match="Repeated execution of the test suite"):
        test("--collect-only")
