import torch
import torchvision.models as tvm
from ..registry import register_cls_model


@register_cls_model("resnet18")
def resnet18(num_classes=1000, pretrained=True):
    model = tvm.resnet18(
        weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


@register_cls_model("efficientnet_b0")
def effnet_b0(num_classes=1000, pretrained=True):
    model = tvm.efficientnet_b0(
        weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, num_classes
    )
    return model
