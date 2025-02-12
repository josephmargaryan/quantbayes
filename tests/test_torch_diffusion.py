# diffusion_lib/test.py

import torch
from quantbayes.torch_based.diffusion import UNet
from quantbayes.torch_based.diffusion import GaussianDiffusion, BetaSchedule


def test_unet_forward():
    """
    Quick sanity check on UNet forward pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=3).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    out = model(x, t)
    print("UNet output shape:", out.shape)


def test_diffusion_step():
    """
    Check if diffusion class can do a forward diffusion and single step back.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=3).to(device)
    beta_schedule = BetaSchedule(schedule_type="linear", timesteps=1000)
    diffusion = GaussianDiffusion(model, beta_schedule)

    x0 = torch.randn(2, 3, 64, 64, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    loss = diffusion.diffusion_loss(x0, t)
    print("Diffusion loss:", loss.item())


if __name__ == "__main__":
    test_unet_forward()
    test_diffusion_step()
