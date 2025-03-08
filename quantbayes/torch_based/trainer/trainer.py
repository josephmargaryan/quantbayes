import torch

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device="cpu",
                 metrics=None, grad_clip=None, amp=False, callbacks=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.metrics = metrics or {}
        self.grad_clip = grad_clip
        self.amp = amp
        self.callbacks = callbacks or []
        self.scaler = torch.amp.GradScaler("cuda") if amp and device == "cuda" else None

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = []
        for batch in dataloader:
            x, y = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            if self.amp:
                with torch.amp.autocast("cuda"):
                    output = self.model(x)
                    loss = self.loss_fn(output, y)
            else:
                output = self.model(x)
                loss = self.loss_fn(output, y)
            loss_val = loss.item()
            epoch_loss.append(loss_val)

            if self.amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            # Optionally update metrics here or via a callback

        if self.scheduler:
            self.scheduler.step()

        # Call callbacks after epoch (logging, checkpointing, etc.)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_loss)

        return sum(epoch_loss) / len(epoch_loss)

    def evaluate(self, dataloader):
        self.model.eval()
        eval_loss = []
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                x, y = [b.to(self.device) for b in batch]
                if self.amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(x)
                else:
                    output = self.model(x)
                loss = self.loss_fn(output, y)
                eval_loss.append(loss.item())
                preds.append(output.cpu())
                targets.append(y.cpu())
        # Calculate custom metrics
        # metrics_result = {name: metric(torch.cat(targets), torch.cat(preds))
        #                   for name, metric in self.metrics.items()}
        return sum(eval_loss) / len(eval_loss)  # , metrics_result

    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=None):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model state, e.g. self.best_model = deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Additional callback triggers can be added here

        return self.model

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset


    # Define a simple linear model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)

    # Create some dummy training and validation data
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 3)
    x_val = torch.randn(20, 10)
    y_val = torch.randn(20, 3)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Instantiate model, loss function, and optimizer
    model = SimpleModel(input_size=10, output_size=3)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create the Trainer instance
    trainer = Trainer(model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=torch.device("cpu"),  # Change to "cuda" if available
                    amp=False)

    # Train the model using the Trainer
    trainer.train(train_loader, val_loader, num_epochs=5, early_stopping_patience=3)
