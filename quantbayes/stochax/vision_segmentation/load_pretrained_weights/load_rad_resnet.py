from pathlib import Path
import numpy as np, torch
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download

########## Only run this if you want to load the pretrained weights from HuggingFace ##########

# 1. grab the checkpoint
pth = Path(hf_hub_download("Lab-Rasool/RadImageNet", "ResNet50.pt", local_dir="."))

# 2. restore & export
state = torch.load(pth, map_location="cpu")
model = resnet50()
model.load_state_dict(state, strict=False)
np.savez(
    "resnet50_radimagenet.npz",
    **{k: v.cpu().numpy() for k, v in model.state_dict().items()},
)
