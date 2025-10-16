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


class TypeMapper:
    @staticmethod
    def map_veeva_type_to_bayes(veeva_type):
        """
        Given a Veeva type string, return the corresponding Bayes type key
        (the first key found in which the value is present). Returns None if not found.
        """
        for bayes_type, veeva_list in type_mapping.items():
            if veeva_type in veeva_list:
                return bayes_type
        return None

    @staticmethod
    def type_similarity(bayes_type, veeva_type):
        """
        Compares the bayes type (expected from the source data) with the veeva type.
        It maps the veeva type using map_veeva_type_to_bayes and returns 1.0 if it equals bayes_type, else 0.0.
        """
        mapped_type = TypeMapper.map_veeva_type_to_bayes(veeva_type)
        return 1.0 if bayes_type == mapped_type else 0.0
