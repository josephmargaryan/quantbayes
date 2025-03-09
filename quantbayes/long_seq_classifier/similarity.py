import numpy as np
import torch
import torch.nn.functional as F


def compute_cosine_similarity(representations):
    """
    Compute cosine similarity between all pairs of document representations.
    """
    normalized_representations = F.normalize(representations, p=2, dim=1)
    similarity_matrix = torch.mm(
        normalized_representations, normalized_representations.t()
    )
    return similarity_matrix


def find_most_similar_documents(similarity_matrix, document_list):
    """
    Identify the two most similar documents.
    """
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    np.fill_diagonal(similarity_matrix_np, -np.inf)  # Ignore self-similarity
    max_sim_indices = np.unravel_index(
        np.argmax(similarity_matrix_np), similarity_matrix_np.shape
    )

    doc1 = document_list[max_sim_indices[0]]
    doc2 = document_list[max_sim_indices[1]]
    similarity_score = similarity_matrix_np[max_sim_indices]

    return doc1, doc2, similarity_score
