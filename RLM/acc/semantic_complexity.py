"""
Semantic Complexity Scorer for the Adaptive Compute Controller (ACC).

Uses Sentence-Transformers to compare a query against "Complexity Prototypes".
This provides a more robust signal than simple keyword matching.
"""

from typing import List, Optional
import torch

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Prototypes representing "Complex" reasoning tasks
_COMPLEX_PROTOTYPES = [
    "multi-hop reasoning across multiple different sources",
    "complex analysis of relationship and intersection between entities",
    "detailed step-by-step logical synthesis of information",
    "identifying the commonality between disparate events or people",
    "reasoning through a logic puzzle with multiple constraints",
    "critiquing and justifying a claim based on evidence",
    "comparing and contrasting multiple complex mechanisms"
]

# Prototypes representing "Simple" fact retrieval
_SIMPLE_PROTOTYPES = [
    "what is the name of this single simple entity",
    "direct fact retrieval and lookup for a specific value",
    "who directed or wrote this one specific work",
    "when was this person born or when did this event occur",
    "list the members or parts of this category",
    "simple definition of a term or concept",
    "where is this location situated"
]

class SemanticComplexityScorer:
    """
    Estimates reasoning complexity using embedding similarity to prototypes.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.fallback_scorer = None
        
        if HAS_SBERT:
            try:
                # Attempt to load the model. 
                # Note: SentenceTransformer often tries to reach HF Hub even for cached models.
                self.model = SentenceTransformer(model_name, device=self.device)
                
                # Pre-encode prototypes
                self.complex_embeddings = self.model.encode(_COMPLEX_PROTOTYPES, convert_to_tensor=True, device=self.device)
                self.simple_embeddings = self.model.encode(_SIMPLE_PROTOTYPES, convert_to_tensor=True, device=self.device)
            except Exception as e:
                # Silently catch and prepare fallback to avoid crashing RLM initialisation
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not initialize SentenceTransformer '{model_name}': {e}. "
                    "Falling back to keyword-based ComplexityScorer."
                )
                self.model = None

        if self.model is None:
            from .complexity import ComplexityScorer
            self.fallback_scorer = ComplexityScorer()

    def score(self, query: str, context: Optional[str] = None) -> float:
        """
        Returns a complexity score in [0, 1].
        1.0 means highly likely to be complex (multi-hop).
        """
        if not query.strip():
            return 0.0
            
        # Use fallback if semantic model failed to load
        if self.model is None:
            return self.fallback_scorer.score(query, context)
            
        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True, device=self.device)
        
        # Calculate similarities
        sim_complex = util.cos_sim(query_emb, self.complex_embeddings).max().item()
        sim_simple = util.cos_sim(query_emb, self.simple_embeddings).max().item()
        
        # Determine relative complexity
        # If it's more similar to simple than complex, keep score low.
        # Score = (sim_complex - sim_simple + 1) / 2
        
        # Margin-based scoring
        score = (sim_complex - sim_simple)
        
        # Normalise margin [-1, 1] to [0, 1]
        # But we want to bias it towards the positive similarity if it's very complex
        # Simple heuristic: 0.5 is neutral.
        normalised_score = (score + 1.0) / 2.0
        
        # Clamp
        final_score = min(max(normalised_score, 0.0), 1.0)
        
        # Context boost (mild)
        if context and len(context) > 1000:
            final_score = min(final_score + 0.1, 1.0)
            
        return round(final_score, 6)
