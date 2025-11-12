import pickle

import pandas as pd
import torch
from dataset import HierarchicalDataset
from model import DocumentClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import (
    generate_wordclouds,
    plot_average_logits_heatmap,
    plot_sentiment_distribution,
    plot_sentiment_polarity,
)


def inference(
    transformer_model,
    classifier_model,
    tokenizer,
    df,
    device,
    label_encoder_path,
    chunk_size=510,
    pooling_strategy="mean",
    batch_size=16,
):
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    transformer_model.to(device)
    classifier_model.to(device)
    transformer_model.eval()
    classifier_model.eval()

    dataloader = DataLoader(
        HierarchicalDataset(
            texts=df["x"].tolist(),
            labels=None,
            tokenizer=tokenizer,
            model=transformer_model,
            device=device,
            chunk_size=chunk_size,
            pooling_strategy=pooling_strategy,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    document_representations = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        document_representations.extend(batch.cpu().numpy())

    embeddings_tensor = torch.tensor(document_representations, device=device)

    with torch.no_grad():
        logits = classifier_model(embeddings_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels, logits.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_data = pd.read_csv("/content/processed_new_data.csv")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transformer_model = AutoModel.from_pretrained("bert-base-uncased")
    hidden_size = transformer_model.config.hidden_size
    num_classes = 5  # Adjust based on your dataset
    classifier_model = DocumentClassifier(hidden_size, num_classes)
    classifier_model.load_state_dict(torch.load("best_model.pth", map_location=device))

    label_encoder_path = "label_encoder.pkl"

    # Perform inference
    predictions, logits = inference(
        transformer_model=transformer_model,
        classifier_model=classifier_model,
        tokenizer=tokenizer,
        df=new_data,
        device=device,
        label_encoder_path=label_encoder_path,
        chunk_size=510,
        pooling_strategy="self_attention",
        batch_size=16,
    )

    # Add predictions to the DataFrame
    new_data["predicted_labels"] = predictions
    new_data["logits"] = [list(l) for l in logits]

    # Save results
    new_data.to_csv("new_data_with_predictions.csv", index=False)
    print("Predictions saved to 'new_data_with_predictions.csv'.")

    # Visualizations
    plot_sentiment_distribution(new_data, label_col="predicted_labels")

    if "company" in new_data.columns:
        plot_average_logits_heatmap(
            new_data, company_col="company", logits_col="logits"
        )

        plot_sentiment_polarity(
            new_data,
            company_col="company",
            logits_col="logits",
            label_col="predicted_labels",
        )

    generate_wordclouds(new_data, text_col="x", label_col="predicted_labels")
