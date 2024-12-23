import torch


def train_epoch(dmm, svi, data_loader, device):
    dmm.train()
    epoch_loss = 0.0
    for padded_sequences, masks, targets in data_loader:
        padded_sequences = padded_sequences.to(device)

        svi_loss = svi.step(padded_sequences)
        epoch_loss += svi_loss
    return epoch_loss / len(data_loader)


def eval_epoch(dmm, svi, data_loader, device):
    dmm.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for padded_sequences, masks, targets in data_loader:
            padded_sequences = padded_sequences.to(device)
            svi_loss = svi.evaluate_loss(padded_sequences)
            epoch_loss += svi_loss
    return epoch_loss / len(data_loader)
