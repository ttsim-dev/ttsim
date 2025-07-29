"""Test the new input_node_paths feature for TTSIM DAG plotting."""

import pytest
from pathlib import Path

from ttsim.plot.dag.tt import tt, NodeSelector


class TestInputNodePaths:
    """Test the input_node_paths feature."""

    def test_basic_plotting_still_works(self):
        """Ensure basic plotting functionality is unchanged."""
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Basic Test",
            include_params=False
        )
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_input_node_paths_parameter(self):
        """Test the new input_node_paths parameter."""
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Test with input_node_paths",
            include_params=False,
            input_node_paths=[("age",)]  # Use actual function name
        )
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_multiple_input_node_paths(self):
        """Test with multiple input node paths."""
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Test with multiple inputs",
            include_params=False,
            input_node_paths=[
                ("age",),
                ("p_id",)
            ]
        )
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_empty_input_node_paths(self):
        """Test with empty input_node_paths list."""
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Test with empty input_node_paths",
            include_params=False,
            input_node_paths=[]
        )
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_node_selector_with_input_node_paths(self):
        """Test combining NodeSelector with input_node_paths."""
        node_selector = NodeSelector(
            node_paths=[("age",)],  # Use actual function, not module
            type="descendants",
            input_node_paths=[("age",)]
        )
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Test NodeSelector with input_node_paths",
            include_params=False,
            node_selector=node_selector
        )
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_input_node_paths_overrides_node_selector(self):
        """Test that input_node_paths parameter overrides NodeSelector's input_node_paths."""
        node_selector = NodeSelector(
            node_paths=[("age",)],  # Use actual function, not module
            type="descendants",
            input_node_paths=[("p_id",)]  # This should be overridden
        )
        fig = tt(
            policy_date_str="2025-01-01",
            root=Path(__file__).parent.parent / "mettsim" / "middle_earth",
            title="Test input_node_paths override",
            include_params=False,
            node_selector=node_selector,
            input_node_paths=[("age",)]  # This should take precedence
        )
        assert fig is not None
        assert hasattr(fig, 'data')
