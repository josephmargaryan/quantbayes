import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from dataset import train_loader, val_loader
from model import DocumentClassifier
import matplotlib.pyplot as plt
import pandas as pd


def train(
    model, num_epochs, train_loader, val_loader, lr, weight_decay, patience, device
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)
    scaler = torch.amp.GradScaler()
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    best_model_state = None
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = []
        for i, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Backpropagating epoch {epoch+1}")
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            avg_train_loss.append(loss.item())
        avg_train_loss = np.mean(avg_train_loss)
        train_losses.append(avg_train_loss)

        avg_val_loss = []
        all_preds = []
        all_labels = []
        model.eval()
        for j, (x, y) in enumerate(tqdm(val_loader, desc=f"Forward pass {epoch+1}")):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(x)
                    loss = criterion(output, y)
                    avg_val_loss.append(loss.item())
                    all_preds.append(torch.argmax(output, dim=1))
                    all_labels.append(y)

        avg_val_loss = np.mean(avg_val_loss)
        val_losses.append(avg_val_loss)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        val_f1 = f1_score(
            all_labels.cpu().numpy(), all_preds.cpu().numpy(), average="weighted"
        )
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        tqdm.write(f"Epoch {epoch+1} | val loss {avg_val_loss:.3f}")
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    if best_model_state is not None:
        torch.save(best_model_state, "best_model.pth")

    df = pd.DataFrame(
        {
            "train": train_losses,
            "val": val_losses,
            "val_accuracy": val_accuracies,
            "val_f1_score": val_f1_scores,
        }
    )
    plt.plot(df.index + 1, df["train"], label="train loss")
    plt.plot(df.index + 1, df["val"], label="val loss")
    plt.plot(df.index + 1, df["val_accuracy"], label="val accuracy")
    plt.plot(df.index + 1, df["val_f1_score"], label="val f1 score")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Loss, Accuracy, and F1 Score over epochs")
    plt.legend()
    plt.savefig("metrics_curve.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentClassifier(hidden_size=768, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)

    train(
        model=model,
        num_epochs=10,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        weight_decay=1e-5,
        patience=3,
        device=device,
    )
