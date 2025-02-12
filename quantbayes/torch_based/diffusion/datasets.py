# diffusion_lib/datasets.py

import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

###############################################################################
# Helper: Center Crop (from ADM)
###############################################################################


def center_crop_arr(pil_image, image_size):
    """
    Center crop an image to the given image_size.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


###############################################################################
# Image Folder Dataset
###############################################################################


class ImageFolderDataset(Dataset):
    """
    Dataset for loading images from a folder.
    """

    def __init__(
        self,
        folder_path,
        image_size=64,
        transform=None,
        extensions=(".png", ".jpg", ".jpeg"),
    ):
        self.folder_path = folder_path
        self.image_size = image_size
        self.extensions = extensions
        self.files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(extensions)
        ]
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Lambda(lambda img: center_crop_arr(img, image_size)),
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3),  # assuming RGB images in [-1,1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


###############################################################################
# Time Series Dataset
###############################################################################


class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series data stored as a numpy array.
    """

    def __init__(self, data_array):
        """
        data_array: numpy array of shape (N, seq_len, D)
        """
        self.data = data_array.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


###############################################################################
# Tabular Dataset
###############################################################################


class TabularDataset(Dataset):
    """
    Dataset for tabular data.
    continuous_data: numpy array (N, num_features)
    categorical_data: optional numpy array (N, num_cat_features)
    """

    def __init__(self, continuous_data, categorical_data=None):
        self.continuous_data = continuous_data.astype(np.float32)
        self.categorical_data = categorical_data
        if categorical_data is not None:
            self.categorical_data = categorical_data.astype(np.int64)

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        if self.categorical_data is not None:
            return self.continuous_data[idx], self.categorical_data[idx]
        else:
            return self.continuous_data[idx]


###############################################################################
# Dummy Dataset for Testing
###############################################################################


class DummyImageDataset(Dataset):
    """
    A dummy image dataset that creates random images.
    """

    def __init__(self, num_samples=100, image_size=64, channels=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a random image tensor in [-1,1]
        return (torch.rand(self.channels, self.image_size, self.image_size) - 0.5) * 2
