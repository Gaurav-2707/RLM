import os
import logging
from typing import Optional
from RLM.utils.llm import LLMClient
from RLM.memory.graph import SemanticContextGraph

logger = logging.getLogger(__name__)

class System2Planner:
    """
    The Graph Recursion Planner.
    Sits before the REPL loop to simulate blast radii using the Semantic Context Graph
    and formulate a multi-step deterministic execution plan.
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.llm = LLMClient()
        self.graph = SemanticContextGraph()
        
        # Build the graph initially
        logger.info(f"Building semantic context graph from {project_root}...")
        self.graph.build_from_directory(project_root)

    def _format_blast_radius(self, target_file: str) -> str:
        """Helper to format the graph search results into a prompt string."""
        rel_path = os.path.relpath(target_file, self.project_root)
        radius = self.graph.get_blast_radius(rel_path, max_depth=2)
        
        if not radius:
            return f"No impacted dependents found for {rel_path} in the semantic graph."
            
        output = [f"Architectural Blast Radius for {rel_path} (impacted dependents):"]
        for node, info in radius.items():
            output.append(f" - [Depth {info['depth']}] {node} ({info['type']})")
        return "\n".join(output)

    def formulate_plan(
        self,
        active_file: str,
        developer_intent: str,
        memory_context: Optional[str] = None,
    ) -> str:
        """
        Uses the LLM and the Semantic Graph to output a JSON step-by-step plan.
        """
        logger.info(f"Formulating System 2 Plan for intent: '{developer_intent}'")
        
        # 1. Graph Recursion to find blast radius
        graph_context = self._format_blast_radius(active_file)
        logger.info(f"Graph Context extracted:\n{graph_context}")
        
        # 2. Construct the prompt
        system_prompt = (
            "You are the System 2 Reasoning Engine for Recursive Labs.\n"
            "Your job is to formulate a deterministic Execution Plan *before* writing code.\n"
            "You will be provided with the developer's intent and a Semantic Graph Blast Radius "
            "showing which files can be impacted by the target file.\n"
            "If episodic memory warnings are provided, avoid repeating those failed approaches.\n"
            "You must output a JSON object containing a list of 'steps', where each step specifies "
            "a file to modify and exactly what to do. Example: \n"
            '{"plan": [{"file": "auth.py", "action": "update jwt"}, {"file": "db.py", "action": "update schema schema"}]}'
        )
        
        user_prompt = (
            f"Developer Intent: {developer_intent}\n\n"
            f"Active File: {active_file}\n\n"
            f"Semantic Graph Context:\n{graph_context}\n\n"
            f"Episodic Memory Context:\n{memory_context or 'No relevant episodic memory found.'}\n\n"
            "Formulate the execution plan now."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Enforce JSON output for deterministic parsing
            response = self.llm.completion(messages, response_format={"type": "json_object"})
            
            # Clean up potential markdown formatting
            from RLM.utils.llm import extract_json_from_text
            clean_json = extract_json_from_text(response)
            
            logger.info("System 2 Plan successfully formulated.")
            return clean_json
            
        except Exception as e:
            logger.error(f"Failed to formulate plan: {e}")
            raise
