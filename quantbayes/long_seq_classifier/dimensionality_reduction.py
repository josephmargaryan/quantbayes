import logging
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap.umap_ as umap
from utils import get_document_representations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_umap(embeddings, n_components=2):
    """
    Apply UMAP dimensionality reduction to the embeddings.
    """
    logging.info(f"Applying UMAP with {n_components} components...")
    reducer = umap.UMAP(n_components=n_components)
    embedding_umap = reducer.fit_transform(embeddings)
    logging.info(f"UMAP transformation complete. Shape: {embedding_umap.shape}")
    return embedding_umap


def plot_embeddings_2d(
    embedding_umap_2d, labels, label_encoder, save_path="umap_2d.png"
):
    """
    Visualize embeddings in 2D using UMAP, colored by their decoded labels.

    Parameters:
    - embedding_umap_2d: 2D UMAP-transformed embeddings.
    - labels: Corresponding target labels (numerical).
    - label_encoder: LabelEncoder object to decode numerical labels to class names.
    - save_path: File path to save the plot.
    """
    decoded_labels = label_encoder.inverse_transform(labels)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_umap_2d[:, 0],
        y=embedding_umap_2d[:, 1],
        hue=decoded_labels,  # Color by decoded labels
        palette="viridis",
        s=100,
        alpha=0.8,
    )
    plt.title("2D UMAP of Document Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Classes")
    plt.savefig(save_path)
    plt.show()


def plot_embeddings_3d(
    embedding_umap_3d, labels, label_encoder, save_path="umap_3d.png"
):
    """
    Visualize embeddings in 3D using UMAP, colored by their decoded labels.

    Parameters:
    - embedding_umap_3d: 3D UMAP-transformed embeddings.
    - labels: Corresponding target labels (numerical).
    - label_encoder: LabelEncoder object to decode numerical labels to class names.
    - save_path: File path to save the plot.
    """
    decoded_labels = label_encoder.inverse_transform(labels)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding_umap_3d[:, 0],
        embedding_umap_3d[:, 1],
        embedding_umap_3d[:, 2],
        c=labels,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )
    ax.set_title("3D UMAP of Document Embeddings")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    fig.colorbar(scatter, ax=ax, label="Classes")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    embeddings, labels = get_document_representations(train_loader, device)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    embedding_umap_2d = apply_umap(embeddings, n_components=2)
    embedding_umap_3d = apply_umap(embeddings, n_components=3)

    plot_embeddings_2d(
        embedding_umap_2d, labels, label_encoder, save_path="umap_train_2d.png"
    )
    plot_embeddings_3d(
        embedding_umap_3d, labels, label_encoder, save_path="umap_train_3d.png"
    )
