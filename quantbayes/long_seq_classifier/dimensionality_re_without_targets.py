import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from dimensionality_reduction import apply_umap


def umap_2d(df, embeddings):
    embedding_umap_2d = apply_umap(embeddings, n_components=2)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_umap_2d[:, 0],
        y=embedding_umap_2d[:, 1],
        hue=df["Company"],
        palette="viridis",
        s=100,
        alpha=0.8,
    )
    plt.title("2D UMAP of News Embeddings Colored by Company")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Company")
    plt.savefig("umap_company_colored.png")
    plt.show()


def umap_3d(df, embeddings):
    embedding_umap_3d = apply_umap(embeddings, n_components=3)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    df["CompanyLabel"] = label_encoder.transform(df["Company"])
    company_labels = df["CompanyLabel"].values

    embedding_umap_3d = apply_umap(embeddings, n_components=3)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        embedding_umap_3d[:, 0],
        embedding_umap_3d[:, 1],
        embedding_umap_3d[:, 2],
        c=company_labels,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )

    ax.set_title("3D UMAP of News Embeddings Colored by Company")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    fig.colorbar(scatter, ax=ax, label="Company")
    plt.savefig("umap_company_colored_3d.png")
    plt.show()
