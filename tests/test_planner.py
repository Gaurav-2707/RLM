import pytest
from unittest.mock import patch
from RLM.engine.planner import System2Planner

@patch("RLM.engine.planner.LLMClient")
def test_system2_planner_blast_radius(mock_llm_client):
    # Mock the LLM completion
    mock_instance = mock_llm_client.return_value
    mock_instance.completion.return_value = '{"plan": [{"file": "tests/fake.py", "action": "fake"}]}'
    
    # Initialize planner (will build graph from RLM root)
    planner = System2Planner(project_root=".")
    
    # Manually add fake nodes to the planner's graph for testing
    planner.graph.add_node("fake_auth.py", "file")
    planner.graph.add_node("fake_db.py", "file")
    planner.graph.add_edge("fake_auth.py", "fake_db.py", "Depends_On")
    
    # Test blast radius string formatting
    context_str = planner._format_blast_radius("fake_db.py")
    assert "fake_auth.py" in context_str
    assert "Depth 1" in context_str
    
    # Test full formulate_plan method
    result = planner.formulate_plan("fake_auth.py", "Fix the bug")
    assert "plan" in result
    
    # Ensure LLM was called
    mock_instance.completion.assert_called_once()
