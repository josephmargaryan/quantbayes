import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(task: str = "segmentation", img_size=512):
    """Returns (train_tf, valid_tf) composed with Albumentations."""
    common = [
        A.Resize(img_size, img_size),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]
    if task == "segmentation":
        train_tf = A.Compose(
            [
                A.Affine(scale=(0.9, 1.1), rotate=15, translate_percent=0.1, p=0.5),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.2),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.RandomBrightnessContrast(0.5),
            ]
            + common,
            additional_targets={"mask": "mask"},
        )
        valid_tf = A.Compose(common, additional_targets={"mask": "mask"})
    else:  # classification
        train_tf = A.Compose(
            [
                A.HorizontalFlip(0.5),
                A.RandomBrightnessContrast(0.2),
            ]
            + common
        )
        valid_tf = A.Compose(common)
    return train_tf, valid_tf
