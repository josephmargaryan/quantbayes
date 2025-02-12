# diffusion_lib/train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from time import time
from quantbayes.torch_based.diffusion import GaussianDiffusion, BetaSchedule

# Simple EMA update function (for one GPU)
def update_ema(ema_model, model, decay=0.9999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# Create a logger that logs both to file and stdout
def create_logger(log_path):
    logger = logging.getLogger("diffusion_train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def train_diffusion_model(
    model,
    diffusion: GaussianDiffusion,
    dataset,
    batch_size=8,
    num_epochs=10,
    lr=1e-4,
    save_path="checkpoints",
    device="cuda",
):
    os.makedirs(save_path, exist_ok=True)
    logger = create_logger(os.path.join(save_path, "train.log"))
    model.to(device)
    diffusion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema_model = (
        type(model)(*model.args) if hasattr(model, "args") else model.__class__()
    )
    ema_model.load_state_dict(model.state_dict())
    ema_model.to(device)
    ema_decay = 0.9999

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader) * num_epochs
    logger.info(
        f"Starting training for {num_epochs} epochs, total steps: {total_steps}"
    )
    start_time = time()

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            # Assume dataset returns a tensor (B, ...) – modify if necessary.
            x = batch.to(device, dtype=torch.float32)
            t = torch.randint(
                0, diffusion.num_timesteps, (x.size(0),), device=device
            ).float()

            loss = diffusion.diffusion_loss(x, t, device=device)
            optimizer.zero_grad()
            loss.backward()
            # Optionally clip gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            update_ema(ema_model, model, decay=ema_decay)
            step += 1

            if step % 100 == 0:
                elapsed = time() - start_time
                logger.info(
                    f"Step {step}/{total_steps}: Loss {loss.item():.4f} - {step/elapsed:.2f} steps/sec"
                )

        # Save checkpoint at end of epoch
        ckpt_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "ema_state": ema_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")
