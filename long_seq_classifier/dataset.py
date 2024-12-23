import torch
from torch.utils.data import Dataset, DataLoader
from utils import apply_pooling
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class HierarchicalDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        model=None,
        device=None,
        chunk_size=510,
        pooling_strategy="mean",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.pooling_strategy = pooling_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels is not None else None
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)

        if tokens.size(0) == 0:
            tokens = torch.tensor([self.tokenizer.unk_token_id])

        chunks = [
            tokens[i : i + (self.chunk_size - 2)]
            for i in range(0, len(tokens), self.chunk_size - 2)
        ]
        padded_chunks = []
        for chunk in chunks:
            chunk = torch.cat(
                [
                    torch.tensor([self.tokenizer.cls_token_id]),
                    chunk,
                    torch.tensor([self.tokenizer.sep_token_id]),
                ]
            )
            padding_length = self.chunk_size - chunk.size(0)
            if padding_length > 0:
                chunk = torch.cat(
                    (chunk, torch.full((padding_length,), self.tokenizer.pad_token_id))
                )
            padded_chunks.append(chunk)

        input_ids = torch.stack(padded_chunks).to(self.device)
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        )

        with torch.no_grad():
            cls_embeddings = []
            for i in range(input_ids.size(0)):
                outputs = self.model(
                    input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                )
                cls_embeddings.append(outputs.last_hidden_state[:, 0, :])
            cls_embeddings = torch.cat(cls_embeddings, dim=0)

        document_representation = apply_pooling(cls_embeddings, self.pooling_strategy)
        return (
            (document_representation, label)
            if label is not None
            else document_representation
        )


def create_dataloader(
    df,
    tokenizer,
    model,
    device,
    batch_size=16,
    chunk_size=510,
    pooling_strategy="mean",
    shuffle=False,
):
    dataset = HierarchicalDataset(
        texts=df["x"].tolist(),
        labels=df.get("y"),
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_size=chunk_size,
        pooling_strategy=pooling_strategy,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("val_data.csv")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    train_loader = create_dataloader(
        df=train_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16,
        chunk_size=510,
        pooling_strategy="self_attention",
        shuffle=True,
    )

    val_loader = create_dataloader(
        df=val_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16,
        chunk_size=510,
        pooling_strategy="self_attention",
        shuffle=False,
    )

    print(f"Train DataLoader has {len(train_loader)} batches.")
    print(f"Validation DataLoader has {len(val_loader)} batches.")
