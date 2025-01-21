import numpy as np
from fuzzywuzzy import fuzz


class SimilarityMetrics:
    @staticmethod
    def levenshtein_similarity(a, b):
        """Levenshtein similarity using fuzzywuzzy."""
        return fuzz.ratio(str(a), str(b)) / 100 if a and b else 0

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Cosine similarity for numeric vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    @staticmethod
    def jaccard_similarity(set_a, set_b):
        """Jaccard similarity for sets."""
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0

    @staticmethod
    def exact_match(a, b):
        """Exact string match."""
        return 1.0 if a == b else 0

    @staticmethod
    def custom_similarity(func, a, b):
        """Allow passing any custom function."""
        return func(a, b) if callable(func) else 0
