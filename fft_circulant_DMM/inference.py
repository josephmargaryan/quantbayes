import torch
import pyro
import pyro.distributions as dist


@torch.no_grad()
def predict_with_uncertainty(dmm, data_loader, device, num_samples=50):
    dmm.eval()
    all_ground_truth = []
    all_means = []
    all_stds = []

    for padded_sequences, masks, targets in data_loader:
        padded_sequences = padded_sequences.to(device)
        loc_seq, scale_seq = dmm.guide_rnn(padded_sequences)  # [B, T, z_dim]

        batch_samples = []
        for _ in range(num_samples):
            # Sample latent states z
            z_samples = dist.Normal(loc_seq, scale_seq).sample()  # [B, T, z_dim]

            preds = []
            for t in range(z_samples.size(1)):
                z_t = z_samples[:, t, :]
                x_loc, x_scale = dmm.emission(z_t)
                # You can also sample from emission distribution if you want more uncertainty
                # emission_sample = dist.Normal(x_loc, x_scale).sample()
                # preds.append(emission_sample)
                # For now, just use mean
                preds.append(x_loc)
            preds = torch.stack(preds, dim=1)  # [B, T, x_dim]
            batch_samples.append(preds)

        batch_samples = torch.stack(batch_samples, dim=0)  # [num_samples, B, T, x_dim]
        mean_preds = batch_samples.mean(dim=0)
        std_preds = batch_samples.std(dim=0)

        all_means.append(mean_preds.cpu().numpy())
        all_stds.append(std_preds.cpu().numpy())
        all_ground_truth.append(padded_sequences.cpu().numpy())

    import numpy as np

    all_means = np.concatenate(all_means, axis=0)
    all_stds = np.concatenate(all_stds, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    return all_ground_truth, all_means, all_stds
