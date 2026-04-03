"""
Adaptive Compute Controller REPL integration.
Combines RLM_REPL with AdaptiveComputeController to dynamically set max iterations.
"""
from typing import Dict, List, Optional, Any
from RLM.rlm_repl import RLM_REPL
from RLM.acc import AdaptiveComputeController, ComplexityScorer

class AdaptiveRLM(RLM_REPL):
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "ollama/llama3",
                 recursive_model: str = "ollama/llama3",
                 enable_logging: bool = False,
                 acc: Optional[AdaptiveComputeController] = None):
        super().__init__(
            api_key=api_key, 
            model=model, 
            recursive_model=recursive_model, 
            enable_logging=enable_logging
        )
        self.acc = acc or AdaptiveComputeController()
        self.scorer = ComplexityScorer()

        # Depth-to-Iterations mapping
        self.depth_to_iters = {1: 5, 2: 10, 3: 20}

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        if query is None:
            query = ""
            
        context_str = str(context) if context else ""
        
        # 1. Compute Complexity
        score = self.scorer.score(query, context=context_str)
        
        # 2. Assign Depth
        depth = self.acc.select_depth(score)
        
        if depth == 0:
            return "Error: Budget Exhausted. Could not run query."
            
        # 3. Modify Iterations Limit dynamically
        self._max_iterations = self.depth_to_iters.get(depth, 5)
        
        # Add basic logging using python's built-in print if logger is not verbose enough
        print(f"[AdaptiveRLM] Assigned Depth {depth} (Score: {score:.4f}) -> {self._max_iterations} max iterations.")
        
        # 4. Standard completion flow
        return super().completion(context, query)
