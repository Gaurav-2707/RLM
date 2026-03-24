import math
import re
from typing import List, Dict, Any

class BM25Retriever:
    """
    Keyword-based retrieval using BM25.
    Optimized for short text fields like 'state' or 'reasoning'.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0
        self.corpus: List[List[str]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace and punctuation tokenization
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit(self, documents: List[str]):
        """Build the BM25 index."""
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.doc_count = len(self.corpus)
        self.df = {}
        total_dl = 0
        
        for tokens in self.corpus:
            total_dl += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        self.avg_dl = total_dl / self.doc_count if self.doc_count > 0 else 0
        
        # Calculate IDF
        for token, freq in self.df.items():
            # BM25 smooth IDF
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str, doc_index: int) -> float:
        """Score a single document against a query."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.corpus[doc_index]
        doc_len = len(doc_tokens)
        
        score = 0.0
        # Calculate token frequency in doc
        tf_dict = {}
        for token in doc_tokens:
            tf_dict[token] = tf_dict.get(token, 0) + 1
            
        for token in query_tokens:
            if token not in self.idf:
                continue
            tf = tf_dict.get(token, 0)
            numerator = self.idf[token] * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
            score += numerator / denominator
            
        return score

class JaccardRetriever:
    """Simple token-based Jaccard similarity retriever."""
    def _tokenize(self, text: str) -> set:
        text = text.lower()
        return set(re.findall(r'\b\w+\b', text))

    def score(self, query: str, document: str) -> float:
        set1 = self._tokenize(query)
        set2 = self._tokenize(document)
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
