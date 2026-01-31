import argparse, torch
from vision import SegmentationPipeline
from vision.data import GenericDataset
from vision.utils import set_seed

"""
Example usage:

from visionlib import SegmentationPipeline, ClassificationPipeline
from visionlib.data import GenericDataset

# ---- segmentation
train_ds = GenericDataset("imgs/", "masks/", task="segmentation")
val_ds   = GenericDataset("imgs_val/", "masks_val/", task="segmentation")

pipe = SegmentationPipeline(["unet_r34","deeplab_v3p_r101"])
pipe.fit(train_ds, val_ds, epochs=50)
pipe.save("seg_ckpt")

# later ...
pipe = SegmentationPipeline.load("seg_ckpt")
pred = pipe.predict(my_numpy_bgr_image)

# ---- classification
cls_tr = GenericDataset("train_imgs/", "train_labels/", task="classification")
cls_val= GenericDataset("val_imgs/", "val_labels/", task="classification")

cls_pipe = ClassificationPipeline(["resnet18"], num_classes=3)
cls_pipe.fit(cls_tr, cls_val, epochs=25)
proba = cls_pipe.predict_proba(torch.from_numpy(batch_images))


"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--archs", nargs="+", default=["unet_r34", "deeplab_v3p_r101"])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    set_seed(42)
    ds_tr = GenericDataset(args.train_dir, args.mask_dir, task="segmentation")
    # 80/20 split
    val_len = int(0.2 * len(ds_tr))
    tr_ds, val_ds = torch.utils.data.random_split(
        ds_tr, [len(ds_tr) - val_len, val_len]
    )
    pipe = SegmentationPipeline(args.archs)
    pipe.fit(tr_ds, val_ds, epochs=args.epochs, save_dir="ckpts_seg")
    pipe.save("seg_pipeline")


if __name__ == "__main__":
    main()
