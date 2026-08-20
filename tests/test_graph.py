import pytest
from RLM.memory.graph import SemanticContextGraph

def test_graph_add_nodes_edges():
    graph = SemanticContextGraph()
    graph.add_node("auth.py", "file")
    graph.add_node("db.py", "file")
    graph.add_edge("auth.py", "db.py", "Depends_On")
    
    assert "auth.py" in graph.graph.nodes
    assert "db.py" in graph.graph.nodes
    assert graph.graph.has_edge("auth.py", "db.py")

def test_blast_radius_bfs():
    graph = SemanticContextGraph()
    
    # Structure:
    # A -> B means A Depends_On B.
    # Blast radius walks reverse edges to find impacted dependents.
    graph.add_node("A", "file")
    graph.add_node("B", "file")
    graph.add_node("C", "file")
    graph.add_node("D", "file")
    
    graph.add_edge("A", "B", "Depends_On")
    graph.add_edge("B", "C", "Depends_On")
    graph.add_edge("A", "D", "Depends_On")
    
    # If C changes, B is impacted at depth 1 and A at depth 2.
    radius = graph.get_blast_radius("C", max_depth=3)
    
    assert "B" in radius
    assert radius["B"]["depth"] == 1
    
    assert "A" in radius
    assert radius["A"]["depth"] == 2
    
    # Ensure source node is not in radius
    assert "C" not in radius

def test_blast_radius_cycle():
    """Test that it doesn't infinite loop on cyclic dependencies."""
    graph = SemanticContextGraph()
    graph.add_node("A", "file")
    graph.add_node("B", "file")
    
    graph.add_edge("A", "B", "Depends_On")
    graph.add_edge("B", "A", "Depends_On")
    
    radius = graph.get_blast_radius("A", max_depth=5)
    assert "B" in radius
    assert len(radius) == 1

def test_graph_persistence_round_trip(tmp_path):
    graph = SemanticContextGraph()
    graph.add_node("auth.py", "file")
    graph.add_node("db.py", "file")
    graph.add_edge("auth.py", "db.py", "Depends_On")

    path = tmp_path / "semantic_graph.json"
    graph.save_to_disk(str(path))

    loaded = SemanticContextGraph()
    loaded.load_from_disk(str(path))

    assert loaded.graph.has_edge("auth.py", "db.py")
    assert loaded.get_blast_radius("db.py")["auth.py"]["depth"] == 1
