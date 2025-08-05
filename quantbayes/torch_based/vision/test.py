import os
import shutil
import tempfile
import json
import numpy as np
import cv2
import torch

from quantbayes.torch_based.vision.data.dataset import GenericDataset
from quantbayes.torch_based.vision.data.transforms import build_transforms
from quantbayes.torch_based.vision.registry import list_seg_models, list_cls_models
from quantbayes.torch_based.vision.pipeline.segmentation_pipeline import (
    SegmentationPipeline,
)
from quantbayes.torch_based.vision.pipeline.classification_pipeline import (
    ClassificationPipeline,
)


def make_synthetic_seg_data(root, n=8, size=(64, 64)):
    """Creates n random RGB images + binary circle masks."""
    img_dir = os.path.join(root, "imgs")
    msk_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    for i in range(n):
        img = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
        mask = np.zeros(size, np.uint8)
        # draw a filled circle
        center = (np.random.randint(16, 48), np.random.randint(16, 48))
        cv2.circle(mask, center, radius=10, color=255, thickness=-1)
        fn = f"{i:03d}.png"
        cv2.imwrite(os.path.join(img_dir, fn), img)
        cv2.imwrite(os.path.join(msk_dir, fn), mask)
    return img_dir, msk_dir


def make_synthetic_cls_data(root, n=8, size=(32, 32), num_classes=2):
    """Creates n random RGB images + .txt labels 0..num_classes-1."""
    img_dir = os.path.join(root, "imgs")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        img = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
        label = np.random.randint(0, num_classes)
        fn = f"{i:03d}.png"
        cv2.imwrite(os.path.join(img_dir, fn), img)
        with open(os.path.join(lbl_dir, f"{i:03d}.txt"), "w") as f:
            f.write(str(label))
    return img_dir, lbl_dir


def test_generic_dataset():
    print("→ Testing GenericDataset (segmentation + inference)...")
    tmp = tempfile.mkdtemp()
    img_dir, msk_dir = make_synthetic_seg_data(tmp, n=4, size=(32, 32))

    # segmentation mode
    ds = GenericDataset(img_dir, msk_dir, task="segmentation", size=(32, 32))
    assert len(ds) == 4, "Length should match"
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape[0] == 3, "Image tensor OK"
    assert isinstance(y, torch.Tensor) and y.shape[0] == 1, "Mask tensor OK"

    # with transforms
    t_train, t_val = build_transforms("segmentation", img_size=32)
    ds.transform = t_train
    x2, y2 = ds[1]
    assert x2.shape == x.shape, "Transformed image same shape"
    assert y2.shape == y.shape, "Transformed mask same shape"

    # inference mode (no mask)
    ds_inf = GenericDataset(img_dir, None, task="segmentation", size=(32, 32))
    xi, yi = ds_inf[2]
    assert yi.numel() == 0, "Inference returns empty target tensor"

    shutil.rmtree(tmp)
    print("  ✓ GenericDataset tests passed.")


def test_generic_dataset_classification():
    print("→ Testing GenericDataset (classification)...")
    tmp = tempfile.mkdtemp()
    img_dir, lbl_dir = make_synthetic_cls_data(tmp, n=6, size=(28, 28), num_classes=3)

    ds = GenericDataset(img_dir, lbl_dir, task="classification", size=(28, 28))
    assert len(ds) == 6
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.ndim == 3, "Image CHW"
    assert isinstance(y, torch.Tensor) and y.dtype == torch.long, "Label tensor long"

    # with transforms
    t_train, t_val = build_transforms("classification", img_size=28)
    ds.transform = t_train
    x2, y2 = ds[1]
    assert x2.shape == x.shape

    shutil.rmtree(tmp)
    print("  ✓ Classification GenericDataset passed.")


def test_registry():
    print("→ Testing registry listing...")
    segs = list_seg_models()
    clss = list_cls_models()
    assert isinstance(segs, list) and len(segs) > 0, "At least one segmentation model"
    assert isinstance(clss, list) and len(clss) > 0, "At least one classification model"
    print(f"  Seg models: {segs[:3]}… total {len(segs)}")
    print(f"  Cls models: {clss[:3]}… total {len(clss)}")
    print("  ✓ Registry tests passed.")


def test_segmentation_pipeline():
    print("→ Testing SegmentationPipeline end-to-end...")
    tmp = tempfile.mkdtemp()
    img_dir, msk_dir = make_synthetic_seg_data(tmp, n=8, size=(32, 32))

    # prepare datasets
    from quantbayes.torch_based.vision.data.dataset import GenericDataset

    ds = GenericDataset(img_dir, msk_dir, task="segmentation", size=(32, 32))
    # split 6/2
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [6, 2])

    # 1-epoch training
    pipe = SegmentationPipeline(["unet_r34"], in_channels=3, num_classes=1)
    pipe.fit(tr_ds, vl_ds, epochs=1, batch_size=2, save_dir=os.path.join(tmp, "ckpt"))
    # predict
    sample = cv2.imread(os.path.join(img_dir, "000.png"))
    pred = pipe.predict(sample)
    assert (
        isinstance(pred, np.ndarray) and pred.shape[:2] == sample.shape[:2]
    ), "Pred shape OK"

    # save & load
    outdir = os.path.join(tmp, "saved_pipe")
    pipe.save(outdir)
    assert os.path.exists(os.path.join(outdir, "meta.json"))
    pipe2 = SegmentationPipeline.load(outdir)
    pred2 = pipe2.predict(sample)
    assert pred2.shape == pred.shape

    # test add/remove
    pipe2.remove_architecture("unet_r34")
    assert "unet_r34" not in pipe2.models
    # re-add
    wpath = os.path.join(outdir, "unet_r34_0.pt")
    pipe2.add_model("unet_r34", wpath)
    assert "unet_r34" in pipe2.models

    shutil.rmtree(tmp)
    print("  ✓ SegmentationPipeline tests passed.")


def test_classification_pipeline():
    print("→ Testing ClassificationPipeline end-to-end...")
    tmp = tempfile.mkdtemp()
    img_dir, lbl_dir = make_synthetic_cls_data(tmp, n=10, size=(28, 28), num_classes=2)

    # prepare datasets
    from quantbayes.torch_based.vision.data.dataset import GenericDataset

    ds = GenericDataset(img_dir, lbl_dir, task="classification", size=(28, 28))
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [8, 2])

    pipe = ClassificationPipeline(["resnet18"], num_classes=2)
    pipe.fit(
        tr_ds, vl_ds, epochs=1, batch_size=4, save_dir=os.path.join(tmp, "ckpt_cls")
    )

    # batch inference
    batch_imgs = []
    for i in range(2):
        im = cv2.imread(os.path.join(img_dir, f"{i:03d}.png"))
        im = cv2.resize(im, (28, 28)).astype(np.float32) / 255.0
        x = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)
        batch_imgs.append(x)
    batch = torch.cat(batch_imgs, 0)
    probs = pipe.predict_proba(batch)
    preds = pipe.predict(batch)
    assert probs.shape == (2, 2)
    assert preds.shape == (2,)

    # save & load
    outdir = os.path.join(tmp, "saved_cls")
    pipe.save(outdir)
    pipe2 = ClassificationPipeline.load(outdir)
    p2 = pipe2.predict(batch)
    assert np.array_equal(p2, preds)

    shutil.rmtree(tmp)
    print("  ✓ ClassificationPipeline tests passed.")


if __name__ == "__main__":
    test_generic_dataset()
    test_generic_dataset_classification()
    test_registry()
    test_segmentation_pipeline()
    test_classification_pipeline()
    print("\nALL TESTS PASSED.")
