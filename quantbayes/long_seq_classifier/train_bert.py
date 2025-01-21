import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

# Hyperparameters
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 510
HEAD_TOKENS = 127
TAIL_TOKENS = 387
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.DataFrame({"Sentence": [...], "Sentiment": [...]})
df["Sentiment"] = df["Sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})


class HeadTailDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, head_tokens, tail_tokens):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]["Sentence"]
        sentiment = self.data.iloc[index]["Sentiment"]

        tokens = self.tokenizer(sentence, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]

        if len(tokens) > self.max_length - 2:
            tokens = tokens[: self.head_tokens] + tokens[-self.tail_tokens :]

        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]

        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(sentiment, dtype=torch.long),
        }


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = HeadTailDataset(df, tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(df["Sentiment"].unique())
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

criterion = nn.CrossEntropyLoss()
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(
        f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "bert_classifier_weights.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
