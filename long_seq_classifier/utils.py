import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import ast


def apply_pooling(cls_embeddings, pooling_strategy="mean"):
    """
    Apply pooling to CLS token embeddings to generate a document representation.

    Parameters:
    - cls_embeddings: Tensor of shape (num_chunks, hidden_size)
    - pooling_strategy: Pooling method ('mean', 'max', 'self_attention')

    Returns:
    - document_representation: Tensor of shape (hidden_size,)
    """
    if pooling_strategy == "mean":
        document_representation = torch.mean(cls_embeddings, dim=0)
    elif pooling_strategy == "max":
        document_representation = torch.max(cls_embeddings, dim=0)[0]
    elif pooling_strategy == "self_attention":
        attn_weights = torch.softmax(
            torch.mm(cls_embeddings, cls_embeddings.transpose(0, 1)), dim=-1
        )
        document_representation = torch.mm(attn_weights, cls_embeddings).mean(0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return document_representation


def get_document_representations(dataloader, device):
    """
    Extract document embeddings using the DataLoader.
    """
    all_representations = []
    all_labels = []
    with torch.no_grad():
        for document_representation, label in tqdm(
            dataloader, desc="Extracting embeddings"
        ):
            all_representations.append(document_representation.to(device))
            all_labels.append(label)

    embeddings = torch.cat(all_representations, dim=0).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return embeddings, labels


def scatter_plot_sentiments(df, logits_column, labels_column, company_column):
    if isinstance(df[logits_column].iloc[0], str):
        df[logits_column] = df[logits_column].apply(ast.literal_eval)

    logits_df = pd.DataFrame(
        df[logits_column].tolist(), columns=["Negative", "Neutral", "Positive"]
    )
    logits_df[labels_column] = df[labels_column]
    logits_df[company_column] = df[company_column]

    fig = px.scatter_3d(
        logits_df,
        x="Negative",
        y="Neutral",
        z="Positive",
        color=company_column,
        symbol=labels_column,
        title="3D Scatter Plot of Sentiments",
    )
    fig.show()


def generate_word_clouds(df, logits_column, news_column, labels_column):
    if isinstance(df[logits_column].iloc[0], str):
        df[logits_column] = df[logits_column].apply(ast.literal_eval)

    for sentiment_class in sorted(df[labels_column].unique()):
        sentiment_news = df[df[labels_column] == sentiment_class][news_column]
        all_text = " ".join(sentiment_news.tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            all_text
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Sentiment Class {sentiment_class}")
        plt.show()
