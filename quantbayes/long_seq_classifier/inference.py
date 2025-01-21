import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from utils import generate_word_clouds, scatter_plot_sentiments

from model import DocumentClassifier
from dataset import HierarchicalDataset


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
    """
    Perform inference on a dataset using the pretrained models.
    """
    # Load the label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    transformer_model.to(device)
    classifier_model.to(device)
    transformer_model.eval()
    classifier_model.eval()

    dataloader = DataLoader(
        HierarchicalDataset(
            texts=df["x"].tolist(),
            labels=[0] * len(df),
            tokenizer=tokenizer,
            model=transformer_model,
            device=device,
            chunk_size=chunk_size,
            pooling_strategy=pooling_strategy,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    all_logits = []
    for batch in tqdm(dataloader, desc="Performing inference"):
        document_representations, _ = batch
        document_representations = document_representations.to(device)

        with torch.no_grad():
            logits = classifier_model(document_representations)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return label_encoder.inverse_transform(all_preds), all_logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transformer_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    hidden_size = transformer_model.config.hidden_size
    num_classes = 3  # Adjust based on your dataset
    classifier_model = DocumentClassifier(hidden_size, num_classes)
    classifier_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    classifier_model.to(device)

    train_df = pd.read_csv("data.csv")
    label_encoder_path = "label_encoder.pkl"

    predictions, logits = inference(
        transformer_model=transformer_model,
        classifier_model=classifier_model,
        tokenizer=tokenizer,
        df=train_df,
        device=device,
        label_encoder_path=label_encoder_path,
        chunk_size=510,
        pooling_strategy="self_attention",
        batch_size=16,
    )

    train_df["predicted_labels"] = predictions
    train_df["logits"] = [list(l) for l in logits]

    train_df.to_csv("train_data_with_predictions.csv", index=False)
    print("Inference complete. Predictions saved to 'data_with_predictions.csv'.")

    print(
        f"Label Encoder Classes: {pickle.load(open(label_encoder_path, 'rb')).classes_}"
    )

    scatter_plot_sentiments(
        data=train_df,
        logits_column="logits",
        labels_column="predicted_labels",
        company_column="Company",
    )
    generate_word_clouds(
        data=train_df,
        logits_column="logits",
        news_column="News",
        labels_column="predicted_labels",
    )
