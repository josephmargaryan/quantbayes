import segmentation_models_pytorch as smp
from ..registry import register_seg_model

# -- UNet / UNet++ --------------------------------------------------------------
@register_seg_model("unet_r34")
def unet_r34(in_channels=3, classes=1, encoder_weights="imagenet"):
    return smp.Unet("resnet34", in_channels=in_channels,
                    classes=classes, encoder_weights=encoder_weights)

@register_seg_model("unetpp_r34")
def unetpp_r34(in_channels=3, classes=1, encoder_weights="imagenet"):
    return smp.UnetPlusPlus("resnet34", in_channels=in_channels,
                            classes=classes, encoder_weights=encoder_weights)

# -- Efficient‑Net encoders -----------------------------------------------------
@register_seg_model("unet_eff_b4")
def unet_eff_b4(in_channels=3, classes=1, encoder_weights="imagenet"):
    return smp.Unet("efficientnet-b4", in_channels=in_channels,
                    classes=classes, encoder_weights=encoder_weights)

@register_seg_model("deeplab_v3p_r101")
def deeplab_v3p_r101(in_channels=3, classes=1, encoder_weights="imagenet"):
    return smp.DeepLabV3Plus("resnet101", in_channels=in_channels,
                             classes=classes, encoder_weights=encoder_weights)

@register_seg_model("linknet_r34")
def linknet_r34(in_channels=3, classes=1, encoder_weights="imagenet"):
    return smp.Linknet("resnet34", in_channels=in_channels,
                       classes=classes, encoder_weights=encoder_weights)
