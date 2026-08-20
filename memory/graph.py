import networkx as nx
import ast
import json
import os
import logging
import sys
import sysconfig
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SemanticContextGraph:
    """
    Horizon 2 Extended Context Graph.
    Replaces purely episodic linear memory with a topological network
    of architectural dependencies to calculate blast radii.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.project_root = None
        self._stdlib_modules = set(getattr(sys, "stdlib_module_names", set()))

    def add_node(self, node_id: str, node_type: str, metadata: Optional[Dict] = None):
        """Adds a node to the context graph."""
        if metadata is None:
            metadata = {}
        self.graph.add_node(node_id, type=node_type, **metadata)

    def add_edge(self, source_id: str, target_id: str, relation_type: str):
        """Adds a directed edge representing an architectural dependency."""
        self.graph.add_edge(source_id, target_id, relation=relation_type)

    def get_blast_radius(self, source_id: str, max_depth: int = 3) -> Dict[str, Dict]:
        """
        Calculates impacted dependents using a reverse Breadth-First Search.

        Depends_On edges are stored as ``consumer -> dependency``. The blast
        radius of a changed dependency is therefore found by walking the graph
        in reverse, returning files/functions that can be affected by a change.
        """
        if source_id not in self.graph:
            return {}

        reverse_graph = self.graph.reverse(copy=False)
        lengths = nx.single_source_shortest_path_length(reverse_graph, source_id, cutoff=max_depth)
        
        blast_radius = {}
        for node, depth in lengths.items():
            if node != source_id: # exclude self
                blast_radius[node] = {
                    "depth": depth,
                    "type": self.graph.nodes[node].get("type", "unknown")
                }
        return blast_radius

    def get_dependencies(self, source_id: str, max_depth: int = 3) -> Dict[str, Dict]:
        """Return upstream dependencies for debugging and planner context."""
        if source_id not in self.graph:
            return {}

        lengths = nx.single_source_shortest_path_length(self.graph, source_id, cutoff=max_depth)
        dependencies = {}
        for node, depth in lengths.items():
            if node != source_id:
                dependencies[node] = {
                    "depth": depth,
                    "type": self.graph.nodes[node].get("type", "unknown"),
                }
        return dependencies

    def _resolve_local_module(self, module_name: str, project_root: str) -> Optional[str]:
        """Map a Python import name to a local file path if it exists."""
        if not module_name:
            return None

        root_name = module_name.split(".")[0]
        if root_name in self._stdlib_modules:
            return None

        module_path = module_name.replace(".", os.sep)
        candidates = [
            os.path.join(project_root, module_path + ".py"),
            os.path.join(project_root, module_path, "__init__.py"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.relpath(candidate, project_root)
        return None

    def parse_python_file_ast(self, filepath: str, project_root: str):
        """
        Parses a python file to automatically extract 'Depends_On' edges 
        based on local imports.
        """
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            try:
                tree = ast.parse(f.read(), filename=filepath)
            except SyntaxError:
                logger.warning(f"Syntax error parsing {filepath}")
                return

        self.project_root = project_root

        # Add the file itself as a node
        rel_filepath = os.path.relpath(filepath, project_root)
        self.add_node(rel_filepath, node_type="file")

        # Find imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = self._resolve_local_module(alias.name, project_root)
                    if target:
                        self.add_node(target, node_type="file")
                        self.add_edge(rel_filepath, target, "Depends_On")
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    target = self._resolve_local_module(node.module, project_root)
                    if target:
                        self.add_node(target, node_type="file")
                        self.add_edge(rel_filepath, target, "Depends_On")

    def build_from_directory(self, directory: str):
        """Recursively builds the graph from all python files in a directory."""
        self.project_root = directory
        for root, _, files in os.walk(directory):
            # Skip virtual environments
            if any(part in {".venv", "__pycache__", ".git"} for part in root.split(os.sep)):
                continue
                
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    self.parse_python_file_ast(full_path, directory)

    def save_to_disk(self, filepath: str):
        """Persist graph nodes and edges to JSON for reproducible planning."""
        data = {
            "project_root": self.project_root,
            "nodes": [
                {"id": node_id, **attrs}
                for node_id, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in self.graph.edges(data=True)
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self, filepath: str):
        """Load a graph saved by save_to_disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self.project_root = data.get("project_root")
        for node in data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        for edge in data.get("edges", []):
            edge = dict(edge)
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
